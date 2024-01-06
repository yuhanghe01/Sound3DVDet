"""
Note: BipartiteMatch Implementation.
"""
import torch
from scipy.optimize import linear_sum_assignment
import torch.nn as nn

torch.autograd.set_detect_anomaly(True)

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """
    def __init__(self, cost_class_weight = 1, cost_pos_weight = 1):
        """Creates the matcher
        Params:
            cost_class_weight: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
        """
        super(HungarianMatcher, self).__init__()
        assert cost_class_weight != 0. or cost_pos_weight != 0., 'the cost weight cannot all be 0'
        self.cost_class_weight = cost_class_weight
        self.cost_pos_weight = cost_pos_weight

    def prepare_data(self, pred_ss3Dpos, gt_ss3Dpos, pred_class_logits, gt_class_labels):
        '''
        Prepare the data for forward processing
        :param pred_ss3Dpos: [N,token_num,3]
        :param gt_ss3Dpos: [N, ss_num, 3]
        :param pred_class_logits: [N, token_num, class_num]
        :param gt_class_labels: [N, ss_num, 1]
        :return: outputs, and targets that fits for forward processing
        '''
        outputs = {'pred_logits': pred_class_logits, 'pred_ss3Dpos': pred_ss3Dpos}

        targets = list()
        batch_size = gt_ss3Dpos.shape[0]

        for batch_id in range(batch_size):
            useful_ss_num = gt_class_labels[batch_id,:] != -1
            useful_ss_num = useful_ss_num.to(torch.int32)
            useful_ss_num = torch.sum(useful_ss_num)
            targets.append({'labels': gt_class_labels[batch_id,0:useful_ss_num],
                            'gt_ss3Dpos': gt_ss3Dpos[batch_id,0:useful_ss_num,:]})

        return outputs, targets

    # @torch.no_grad()
    # def forward(self, outputs, targets):
    @torch.no_grad()
    def forward(self, pred_ss3Dpos, gt_ss3Dpos, pred_class_logits, gt_class_labels):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        outputs, targets = self.prepare_data(pred_ss3Dpos, gt_ss3Dpos, pred_class_logits, gt_class_labels)

        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_ss3Dpos = outputs["pred_ss3Dpos"].flatten(0, 1)  # [batch_size * num_queries, 3]

        # Also concat the target labels and ss3Dpos
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_ss3Dpos = torch.cat([v["gt_ss3Dpos"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids] #[688, 126]
        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_ss3Dpos, tgt_ss3Dpos, p=1) #[688, 126]

        # Final cost matrix
        C = self.cost_pos_weight * cost_bbox + self.cost_class_weight * cost_class
        C = C.view(bs, num_queries, -1).cpu() #[43, 16, 126]

        sizes = [len(v["labels"]) for v in targets]
        # with torch.no_grad():
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

if __name__ == '__main__':
    hungarian_matcher = HungarianMatcher()
    import numpy as np
    import os
    os.putenv('CUDA_VISISBLE_DEVICES', '0')
    device = torch.device('cuda:0')
    query_num = 10
    batch_size = 3
    class_num = 5

    pred_class_logits = np.random.rand(batch_size, query_num, class_num)
    pred_class_logits = torch.from_numpy(pred_class_logits).to(device)
    pred_ss_3Dpos = np.random.rand(batch_size, query_num, 3)
    pred_ss_3Dpos = torch.from_numpy(pred_ss_3Dpos).to(device)

    gt_class_labels = list()
    gt_ss_3D_pos = list()

    gt_class_labels.append(torch.from_numpy(np.array([0,1,2], np.int32)).to(device).to(torch.long))
    gt_class_labels.append(torch.from_numpy(np.array([1,2], np.int32)).to(device).to(torch.long))
    gt_class_labels.append(torch.from_numpy(np.array([3], np.int32)).to(device).to(torch.long))

    gt_ss_3D_pos.append(torch.from_numpy(np.random.rand(3,3)).to(device))
    gt_ss_3D_pos.append(torch.from_numpy(np.random.rand(2,3)).to(device))
    gt_ss_3D_pos.append(torch.from_numpy(np.random.rand(1,3)).to(device))

    outputs = {'pred_logits': pred_class_logits, 'pred_boxes': pred_ss_3Dpos}

    targets = list()
    targets.append({'labels': gt_class_labels[0], 'boxes': gt_ss_3D_pos[0]})
    targets.append({'labels': gt_class_labels[1], 'boxes': gt_ss_3D_pos[1]})
    targets.append({'labels': gt_class_labels[2], 'boxes': gt_ss_3D_pos[2]})

    pred_idx = hungarian_matcher(outputs, targets)

    breakpoint()


