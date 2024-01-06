'''
Note: various coordinate system transform function. For example from camera coordinate system to 3D world coordinate system,
        from camera coordinate system 1 to camera coordinate system 2, etc
Author: Sangyun, Yuhang He
Email: yuhanghe01@gmail.com
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

import rigid_body_motion as rbm
import quaternion
import numpy as np


"""
Pytorch based implementation
"""

class CoordTransform(nn.Module):
    def __init__(self, device = 'cpu:0',
                 imgplane_height=512,
                 imgplane_width=512):
        super(CoordTransform, self).__init__()
        self.device = device
        self.imgplane_height = imgplane_height
        self.imgplane_width = imgplane_width

    def get_world_from_camera_torch(self, points_in_cam, cam_pos, cam_rot):
        #--Input--
        #point_in_cam : (NX3) torch tensor
        #cam_pos : (,3) numpy array
        #cam_rot : rotation in quaternion object from simulation

        #--Output--
        #point_in_world : (1X3) torch tensor
        rf_world = rbm.ReferenceFrame("world")

        rf_observer = rbm.ReferenceFrame(
            "observer",
            parent=rf_world,
            translation=(cam_pos[0].detach().cpu().numpy(),
                         cam_pos[1].detach().cpu().numpy(),
                         cam_pos[2].detach().cpu().numpy()),
            rotation=(cam_rot[1].detach().cpu().numpy(), cam_rot[2].detach().cpu().numpy(),
                      cam_rot[3].detach().cpu().numpy(), cam_rot[0].detach().cpu().numpy()),
        )

        rf_world.register()
        rf_observer.register()

        quat_from_rbm = rbm.lookup_transform(outof='observer',into='world')[1]
        trans_from_rbm = rbm.lookup_transform(outof='observer',into='world')[0]

        point_in_world = self.rigid_transform_torch(points_in_cam, torch.tensor(quat_from_rbm)[None,:], torch.tensor(trans_from_rbm)[None,:])

        rbm.deregister_frame("world")
        rbm.deregister_frame("observer")

        return point_in_world


    def get_camera_from_world_torch(self, points_in_world, cam_pos, cam_rot):
        #--Input--
        #point_in_world: (NX3) torch tensor
        #cam_pos : (,3) numpy array
        #cam_rot : rotation in quaternion object from simulation

        #--Output--
        #point_in_camera : (1X3) torch tensor
        rf_world = rbm.ReferenceFrame("world")
        rf_observer = rbm.ReferenceFrame(
            "observer",
            parent=rf_world,
            translation=(cam_pos[0].detach().cpu().numpy(),
                         cam_pos[1].detach().cpu().numpy(),
                         cam_pos[2].detach().cpu().numpy()),
            rotation=(cam_rot[1].detach().cpu().numpy(),
                      cam_rot[2].detach().cpu().numpy(),
                      cam_rot[3].detach().cpu().numpy(),
                      cam_rot[0].detach().cpu().numpy()),
        )
        rf_world.register()
        rf_observer.register()

        quat_from_rbm = rbm.lookup_transform(outof='world',into='observer')[1]
        quat_from_rbm = quat_from_rbm.astype(np.float32)
        trans_from_rbm = rbm.lookup_transform(outof='world',into='observer')[0]
        trans_from_rbm = trans_from_rbm.astype(np.float32)

        point_in_camera = self.rigid_transform_torch(points_in_world,
                                                     torch.tensor(quat_from_rbm).to(self.device)[None,:],
                                                     torch.tensor(trans_from_rbm).to(self.device)[None,:])
        rbm.deregister_frame("world")
        rbm.deregister_frame("observer")

        return point_in_camera

    def rigid_transform_torch(self, point_cloud, rot, translation):
        #point_cloud : N X 3 torch tensor
        #rot : 1 X 4 quaternion wxyz
        #translation : 1 X 3 translation

        rot_mat_rbm = self.quat2rot(rot)
        transformed = (point_cloud @ rot_mat_rbm[0].to(torch.float32).to(point_cloud.device)) + \
                      translation[0].to(torch.float32).to(point_cloud.device)

        return transformed

    def quat2rot(self, quat):
        #quat : N X 4 (wxyz)
        q11 = 2*(quat[:,0]**2+quat[:,1]**2) - 1
        q12 = 2*(quat[:,1]*quat[:,2] - quat[:,0]*quat[:,3])
        q13 = 2*(quat[:,1]*quat[:,3] + quat[:,0]*quat[:,2])

        q21 = 2*(quat[:,1]*quat[:,2] + quat[:,0]*quat[:,3])
        q22 = 2*(quat[:,0]**2+quat[:,2]**2) - 1
        q23 = 2*(quat[:,2]*quat[:,3] - quat[:,0]*quat[:,1])

        q31 = 2*(quat[:,1]*quat[:,3] - quat[:,0]*quat[:,2])
        q32 = 2*(quat[:,2]*quat[:,3] - quat[:,0]*quat[:,1])
        q33 = 2*(quat[:,0]**2+quat[:,3]**2) - 1

        row1 = torch.cat([q11[:,None],q12[:,None],q13[:,None]],1)
        row2 = torch.cat([q21[:,None],q22[:,None],q23[:,None]],1)
        row3 = torch.cat([q31[:,None],q32[:,None],q33[:,None]],1)

        rot_mat = torch.cat([row1[:,None,:],row2[:,None,:],row3[:,None,:]],1)

        return rot_mat


    def transform_cameraA_to_cameraB(self, point_cameraA, cameraA_pos, cameraA_rot,
                                     cameraB_pos, cameraB_rot):
        '''
        transform 3D point in camera coordinate A to camera coordinate B.
        :param point_cameraA: [N, 3], 3D points expressed in camera coord A
        :param cameraA_pos:
        :param cameraA_rot:
        :param cameraB_pos:
        :param cameraB_rot:
        :return: [N, 3] camera expressed point in Camera Coord B
        '''
        point_worldCoord = self.get_world_from_camera_torch(point_cameraA,
                                                            cam_pos=cameraA_pos,
                                                            cam_rot=cameraA_rot)

        point_cameraB = self.get_camera_from_world_torch(point_worldCoord,
                                                         cam_pos=cameraB_pos,
                                                         cam_rot=cameraB_rot)

        return point_cameraB


    def project_cameraCoord_point_to_imgPlane(self, cameraCoord_pos3d):
        '''
        ----Input----
        cameraCoord_pos3d: N X 3 (torch float tensor)

        ----Output----
        pixel_locs: N X 2 (torch float tensor)
        is_inside: N (torch bool tensor)
        '''
        #######Fixed parameters
        fx,fy = 256,256
        cx,cy = 256,256
        height, width = 512,512
        #######

        zz = cameraCoord_pos3d[:,2]
        pixel_loc_x = ((fx*cameraCoord_pos3d[:,0])/(zz+1e-16)) + cx
        pixel_loc_y = ((fy*cameraCoord_pos3d[:,1])/(zz+1e-16)) + cy

        pixel_locs = torch.cat([pixel_loc_x[:,None], pixel_loc_y[:,None]],1)

        is_inside = (pixel_loc_x >= 0) * (pixel_loc_x < width) * (pixel_loc_y >= 0) * (pixel_loc_y < height)

        #explicitly normalise the pixel_locs between [-1, 1], due to the following F.grid_sampling function requirement
        pixel_locs[:, 0] = (pixel_locs[:, 0] - self.imgplane_height / 2.) / self.imgplane_height
        pixel_locs[:, 1] = (pixel_locs[:, 1] - self.imgplane_width / 2.) / self.imgplane_width

        return pixel_locs, is_inside

"""
the corresponding numpy based implementation
"""

import DataProvider

if __name__ == '__main__':
    seq_dataset = DataProvider.Sound3DVDetDataset()

    data_loader = torch.utils.data.DataLoader(seq_dataset,
                                              batch_size=2,
                                              shuffle=False,)

    coord_transformer = CoordTransform()

    for data_tmp in data_loader:
        rgb_feat, gccphat_feat, ss_pos, cam_pos, cam_rot = data_tmp

        cameraA_points = ss_pos[0,0,:,:]
        cameraA_pos = cam_pos[0,0,:]
        cameraA_rot = cam_rot[0,0,:]

        cameraB_pos = cam_pos[0,2,:]
        cameraB_rot = cam_rot[0,2,:]

        cameraB_points = coord_transformer.transform_cameraA_to_cameraB(cameraA_points,
                                                                        cameraA_pos,
                                                                        cameraA_rot,
                                                                        cameraB_pos,
                                                                        cameraB_rot)

        breakpoint()





