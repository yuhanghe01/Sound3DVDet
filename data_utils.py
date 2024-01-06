import torch
import numpy as np
import torch.nn as nn
import rigid_body_motion as rbm

class DataProcessor:
    def __init__(self):
        pass

    def get_3D_pos_from_2D(self, px, py, cam_pose, depth):
        fx,fy = 256,256
        cx,cy = 256,256
        height, width = 512,512

        zz = depth[py][px]
        xx = (px - cx) * zz / fx
        yy = (py - cy) * zz / fy

        cam_pos, cam_rot = cam_pose[0], cam_pose[1]
        pos_3d_in_cam = np.array([xx[0],yy[0],zz[0]])[None,:]

        rf_world = rbm.ReferenceFrame("world")
        rf_observer = rbm.ReferenceFrame(
            "observer",
            parent=rf_world,
            translation=(cam_pos[0], cam_pos[1], cam_pos[2]),
            rotation=(cam_rot.x, cam_rot.y,
                      cam_rot.z, cam_rot.w),
        )
        rf_world.register()
        rf_observer.register()

        pos_3d_in_world = rbm.transform_points(pos_3d_in_cam, outof="observer", into="world")
        rbm.deregister_frame("world")
        rbm.deregister_frame("observer")

        return pos_3d_in_world, pos_3d_in_cam

    def get_2D_pos_from_absolute_3D_rotasarray(self, position_3d, cam_pose):
        #######Fixed parameters
        fx,fy = 256,256
        cx,cy = 256,256
        height, width = 512,512
        #######

        cam_pos, cam_rot = cam_pose[0], cam_pose[1]

        '''Camera rot is saved as
        rot_array[0] = agent_rot.w
        rot_array[1] = agent_rot.x
        rot_array[2] = agent_rot.y
        rot_array[3] = agent_rot.z
        '''

        rf_world = rbm.ReferenceFrame("world")
        rf_observer = rbm.ReferenceFrame(
            "observer",
            parent=rf_world,
            translation=(cam_pos[0], cam_pos[1], cam_pos[2]),
            rotation=(cam_rot[1], cam_rot[2],
                      cam_rot[3], cam_rot[0]),
        )
        rf_world.register()
        rf_observer.register()

        pos_3d_in_ref = rbm.transform_points(position_3d[None,:], outof="world", into="observer")

        zz = pos_3d_in_ref[0,2]
        pixel_loc_x = ((fx*pos_3d_in_ref[0,0])/(zz+1e-16)) + cx
        pixel_loc_y = ((fy*pos_3d_in_ref[0,1])/(zz+1e-16)) + cy

        if pixel_loc_x >= 0 and pixel_loc_x < width and pixel_loc_y >= 0 and pixel_loc_y < height:
            is_inside = True
        else:
            is_inside = False
        rbm.deregister_frame("world")
        rbm.deregister_frame("observer")

        return [pixel_loc_x, pixel_loc_y], is_inside

    def get_2D_pos_from_absolute_3D(self, position_3d, cam_pose, depth):
        #######Fixed parameters
        fx,fy = 256,256
        cx,cy = 256,256
        height, width = 512,512
        #######

        cam_pos, cam_rot = cam_pose[0], cam_pose[1]
        rf_world = rbm.ReferenceFrame("world")
        rf_observer = rbm.ReferenceFrame(
            "observer",
            parent=rf_world,
            translation=(cam_pos[0], cam_pos[1], cam_pos[2]),
            rotation=(cam_rot.x, cam_rot.y,
                      cam_rot.z, cam_rot.w),
        )
        rf_world.register()
        rf_observer.register()

        pos_3d_in_ref = rbm.transform_points(position_3d[None,:], outof="world", into="observer")

        zz = pos_3d_in_ref[0,2]
        pixel_loc_x = ((fx*pos_3d_in_ref[0,0])/(zz+1e-16)) + cx
        pixel_loc_y = ((fy*pos_3d_in_ref[0,1])/(zz+1e-16)) + cy

        if pixel_loc_x >= 0 and pixel_loc_x < width and pixel_loc_y >= 0 and pixel_loc_y < height:
            is_inside = True
        else:
            is_inside = False
        rbm.deregister_frame("world")
        rbm.deregister_frame("observer")

        return [pixel_loc_x, pixel_loc_y], is_inside

    def camera2world_transform(self, pos_3d_in_cam, cam_pose ):
        cam_pos, cam_rot = cam_pose[0], cam_pose[1]

        rf_world = rbm.ReferenceFrame("world")
        rf_observer = rbm.ReferenceFrame(
            "observer",
            parent=rf_world,
            translation=(cam_pos[0], cam_pos[1], cam_pos[2]),
            rotation=(cam_rot.x, cam_rot.y,
                      cam_rot.z, cam_rot.w),
        )

        rf_world.register()
        rf_observer.register()

        pos_3d_in_world = rbm.transform_points(pos_3d_in_cam, outof="observer", into="world")
        rbm.deregister_frame("world")
        rbm.deregister_frame("observer")

        return pos_3d_in_world

    def world2camera_transform_rotasarray(self, pos_3d_in_world, cam_pose):
        '''
        Need Sangyun's help to implement it.
        '''
        cam_pos, cam_rot = cam_pose[0], cam_pose[1]

        #cam_rot is an array. saved as:
        '''
        rot_array[0] = agent_rot.w
        rot_array[1] = agent_rot.x
        rot_array[2] = agent_rot.y
        rot_array[3] = agent_rot.z
        '''

        rf_world = rbm.ReferenceFrame("world")
        rf_observer = rbm.ReferenceFrame(
            "observer",
            parent=rf_world,
            translation=(cam_pos[0], cam_pos[1], cam_pos[2]),
            rotation=(cam_rot[1], cam_rot[2],
                      cam_rot[3], cam_rot[0]),
        )

        rf_world.register()
        rf_observer.register()

        pos_3d_in_cam = rbm.transform_points(pos_3d_in_world, outof="world", into="observer")
        rbm.deregister_frame("world")
        rbm.deregister_frame("observer")

        return pos_3d_in_cam

    def world2camera_transform(self, pos_3d_in_world, cam_pose):
        '''
        Need Sangyun's help to implement it.
        '''
        breakpoint()
        cam_pos, cam_rot = cam_pose[0], cam_pose[1]

        rf_world = rbm.ReferenceFrame("world")
        rf_observer = rbm.ReferenceFrame(
            "observer",
            parent=rf_world,
            translation=(cam_pos[0], cam_pos[1], cam_pos[2]),
            rotation=(cam_rot.x, cam_rot.y,
                      cam_rot.z, cam_rot.w),
        )

        rf_world.register()
        rf_observer.register()

        pos_3d_in_cam = rbm.transform_points(pos_3d_in_world, outof="world", into="observer")
        rbm.deregister_frame("world")
        rbm.deregister_frame("observer")

        return pos_3d_in_cam


    def get_grid_center_coords(self,
                               grid_cell_size=0.01,
                               grid_map_size = 5.12,
                               depth_shift = 0.3,
                               y_init_val = 0.5):
        '''
        We explicitly create a 2D horizontal grid map, with grid_resolution, and grid_size predifined. In our case, the x-, y-
        axis are lying on the RGB image plane, while the z- axis lies in forward-looking direction
        :param grid_cell_size: grid cell size, we take it as 0.1
        :param grid_map_size: grid map size
        :param depth_shift: to guarantee the sound source doesn't lie too close to the RGB image, we explicitly set a
            depth shift
        :return: [H, W, 3], usually H=W, 3 means the three channels mean centered [x, y, z] coordinate
        '''
        cell_num = int(grid_map_size/grid_cell_size)
        # x_range = np.arange(cell_num).astype(np.float32) - cell_num//2
        # # x_coord = x_range * grid_cell_size

        z_min = depth_shift
        z_max = depth_shift + grid_map_size

        x_min = -1. * z_max
        x_max = z_max

        x_coords = np.arange(x_min, x_max, step=grid_cell_size*2)
        x_coord_shift = (x_coords.shape[0] - cell_num)//2
        x_coords = x_coords[x_coord_shift:x_coord_shift+cell_num]
        x_coords = np.expand_dims(x_coords, axis=0)
        x_coords = np.tile(x_coords, reps=[cell_num, 1])

        z_coords = np.arange(z_min, z_max, step=grid_cell_size)
        z_coords = np.expand_dims(z_coords, axis=1)
        z_coords = np.tile(z_coords, reps=[1,cell_num])

        y_coords = np.ones([cell_num], np.float32)*y_init_val
        y_coords = np.expand_dims(y_coords, axis=0)
        y_coords = np.tile(y_coords, reps=[cell_num, 1])

        grid_init = np.stack([x_coords, y_coords, z_coords], axis=2)

        #most of the grids are useless, we create a mask to record each cells usefullness

        return grid_init

    def localize_ss_gridmap(self, ss_list, grid_map):
        grid_map_xz = np.stack([grid_map[:,:,0], grid_map[:,:,2]], axis=-1)

        ss_pos_list = list()
        pos_shift_list = list()
        for ss_pos in ss_list:
            x, z = ss_pos[0], ss_pos[2]
            x = np.ones([grid_map.shape[0], grid_map.shape[1]], dtype=np.float32)*x
            z = np.ones([grid_map.shape[0], grid_map.shape[1]], dtype=np.float32)*z
            xz_map = np.stack([x, z], axis=-1)
            dist_map = np.sqrt(np.sum(np.square(xz_map - grid_map_xz), axis=2, keepdims=False))
            pos_row = np.where(dist_map == np.min(dist_map))[0][0]
            pos_col = np.where(dist_map == np.min(dist_map))[1][0]

            anchor_pos =  grid_map[pos_row, pos_col,:]
            pos_shift = ss_pos - anchor_pos

            ss_pos_list.append([pos_row, pos_col])
            pos_shift_list.append(pos_shift)

        ss_pos = np.array(ss_pos_list, dtype=np.float32)
        pos_shift = np.array(pos_shift_list, dtype=np.float32)

        return ss_pos, pos_shift






if __name__ == '__main__':
    data_processor = DataProcessor()

    data_processor.get_grid_center_coords()


