import torch
import torch.nn as nn
import torch.nn.functional as F
import BipartiteMatch

def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes

class SSDetLoss(nn.Module):
    """ This class computes the loss for Sound3DVDet.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, eos_coef,
                 cost_class_weight = 1., cost_pos_weight = 1.,
                 deeply_supervise = True,
                 multiview_num = 10,
                 transformer_layer_num = 6):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: bipartite matching module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super(SSDetLoss, self).__init__()
        self.num_classes = num_classes
        self.matcher = BipartiteMatch.HungarianMatcher(cost_class_weight = cost_class_weight,
                                                       cost_pos_weight = cost_pos_weight)
        # self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.cost_class_weight = cost_class_weight
        self.cost_pos_weight = cost_pos_weight
        self.deeply_supervise = deeply_supervise
        self.multiview_num = multiview_num
        self.transformer_layer_num = transformer_layer_num
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)

        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        # losses = {'loss_ce': loss_ce}

        return loss_ce

    def loss_ss_3Dpos(self, outputs, target, indices, num_ss):
        """Compute the losses between predicted sound source 3D pos and ground truth sound source 3D pos.
        with L1 loss is used.
        """
        assert 'pred_ss3Dpos' in outputs
        idx = self._get_src_permutation_idx(indices)
        pred_ss_3Dpos = outputs['pred_ss3Dpos'][idx]
        gt_ss_3Dpos = torch.cat([t['gt_ss3Dpos'][i] for t, (_, i) in zip(target, indices)], dim=0)

        loss_ss_3Dpos = F.l1_loss(pred_ss_3Dpos, gt_ss_3Dpos, reduction='none')

        loss_ss_3Dpos = loss_ss_3Dpos.sum()/num_ss

        return loss_ss_3Dpos

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])

        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def compute_oneset_loss(self, pred_ss3Dpos, gt_ss3Dpos, pred_class_logits, gt_class_labels):
        match_indices = self.matcher(pred_ss3Dpos=pred_ss3Dpos,
                                     gt_ss3Dpos=gt_ss3Dpos,
                                     pred_class_logits=pred_class_logits,
                                     gt_class_labels=gt_class_labels)

        outputs, targets = self.matcher.prepare_data(pred_ss3Dpos, gt_ss3Dpos, pred_class_logits, gt_class_labels)


        ce_loss = self.loss_labels(outputs, targets, match_indices)

        num_ss = sum(len(t["labels"]) for t in targets)
        num_ss = torch.as_tensor([num_ss], dtype=torch.float, device=next(iter(outputs.values())).device)

        regress_loss = self.loss_ss_3Dpos(outputs, targets, match_indices, num_ss=num_ss)
        loss = self.cost_class_weight * ce_loss + self.cost_pos_weight * regress_loss

        return loss

    def forward(self, input_dict):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        loss_sound3dvdet = self.compute_oneset_loss(pred_ss3Dpos=input_dict['sound3dvdet_output']['pred_ss3Dpos'],
                                     gt_ss3Dpos=input_dict['sound3dvdet_output']['gt_ss3Dpos'],
                                     pred_class_logits=input_dict['sound3dvdet_output']['pred_ssclasslogits'],
                                     gt_class_labels=input_dict['sound3dvdet_output']['gt_ssclasslabels'])

        if self.deeply_supervise:
            loss_refview = self.compute_oneset_loss(pred_ss3Dpos=input_dict['refview_output']['pred_ss3Dpos'],
                                     gt_ss3Dpos=input_dict['refview_output']['gt_ss3Dpos'],
                                     pred_class_logits=input_dict['refview_output']['pred_ssclasslogits'],
                                     gt_class_labels=input_dict['refview_output']['gt_ssclasslabels'])

            loss_multiview = 0.

            for view_id in range(self.multiview_num-1):
                loss_multiview_tmp = self.compute_oneset_loss(pred_ss3Dpos=input_dict['multiview_output']['view_{}'.format(view_id)]['pred_ss3Dpos'],
                                     gt_ss3Dpos=input_dict['multiview_output']['view_{}'.format(view_id)]['gt_ss3Dpos'],
                                     pred_class_logits=input_dict['multiview_output']['view_{}'.format(view_id)]['pred_ssclasslogits'],
                                     gt_class_labels=input_dict['multiview_output']['view_{}'.format(view_id)]['gt_ssclasslabels'])
                loss_multiview += loss_multiview_tmp

            loss_transformlayers = 0.
            for layer_id in range(self.transformer_layer_num):
                loss_translayer_tmp = self.compute_oneset_loss(pred_ss3Dpos=input_dict['transformer_interm_output']['layer_{}'.format(layer_id)]['pred_ss3Dpos'],
                                     gt_ss3Dpos=input_dict['transformer_interm_output']['layer_{}'.format(layer_id)]['gt_ss3Dpos'],
                                     pred_class_logits=input_dict['transformer_interm_output']['layer_{}'.format(layer_id)]['pred_ssclasslogits'],
                                     gt_class_labels=input_dict['transformer_interm_output']['layer_{}'.format(layer_id)]['gt_ssclasslabels'])
                loss_transformlayers += loss_translayer_tmp

            print('loss_sound3dvdet = {}, loss_refview = {},'
                  'loss_multiview = {}, loss_transformlayer = {}'.format(loss_sound3dvdet,
                                                                         loss_refview,
                                                                         loss_multiview,
                                                                         loss_transformlayers))

            return loss_sound3dvdet + loss_refview + loss_multiview + loss_transformlayers

        else:
            print('loss_sound3dvdet = {}'.format(loss_sound3dvdet))
            return loss_sound3dvdet

    # def forward_bak(self, outputs, targets):
    #     """ This performs the loss computation.
    #     Parameters:
    #          outputs: dict of tensors, see the output specification of the model for the format
    #          targets: list of dicts, such that len(targets) == batch_size.
    #                   The expected keys in each dict depends on the losses applied, see each loss' doc
    #     """
    #     #step1: compute transformer prediction
    #     outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
    #
    #     # Retrieve the matching between the outputs of the last layer and the targets
    #     indices = self.matcher(outputs_without_aux, targets)
    #
    #     # Compute the average number of target boxes accross all nodes, for normalization purposes
    #     num_boxes = sum(len(t["labels"]) for t in targets)
    #     num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
    #     if is_dist_avail_and_initialized():
    #         torch.distributed.all_reduce(num_boxes)
    #     num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
    #
    #     # Compute all the requested losses
    #     losses = {}
    #     for loss in self.losses:
    #         losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))
    #
    #     # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
    #     if 'aux_outputs' in outputs:
    #         for i, aux_outputs in enumerate(outputs['aux_outputs']):
    #             indices = self.matcher(aux_outputs, targets)
    #             for loss in self.losses:
    #                 if loss == 'masks':
    #                     # Intermediate masks losses are too costly to compute, we ignore them.
    #                     continue
    #                 kwargs = {}
    #                 if loss == 'labels':
    #                     # Logging is enabled only for the last layer
    #                     kwargs = {'log': False}
    #                 l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
    #                 l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
    #                 losses.update(l_dict)
    #
    #     return losses