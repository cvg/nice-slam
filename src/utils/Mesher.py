import numpy as np
import open3d as o3d
import skimage
import torch
import torch.nn.functional as F
import trimesh
from packaging import version
from src.utils.datasets import get_dataset


class Mesher(object):

    def __init__(self, cfg, args, slam, points_batch_size=500000, ray_batch_size=100000):
        """
        Mesher class, given a scene representation, the mesher extracts the mesh from it.

        Args:
            cfg (dict): parsed config dict.
            args (class 'argparse.Namespace'): argparse arguments.
            slam (class NICE-SLAM): NICE-SLAM main class.
            points_batch_size (int): maximum points size for query in one batch. 
                                     Used to alleviate GPU memeory usage. Defaults to 500000.
            ray_batch_size (int): maximum ray size for query in one batch. 
                                  Used to alleviate GPU memeory usage. Defaults to 100000.
        """
        self.points_batch_size = points_batch_size
        self.ray_batch_size = ray_batch_size
        self.renderer = slam.renderer
        self.coarse = cfg['coarse']
        self.scale = cfg['scale']
        self.occupancy = cfg['occupancy']
        
        self.resolution = cfg['meshing']['resolution']
        self.level_set = cfg['meshing']['level_set']
        self.clean_mesh_bound_scale = cfg['meshing']['clean_mesh_bound_scale']
        self.remove_small_geometry_threshold = cfg['meshing']['remove_small_geometry_threshold']
        self.color_mesh_extraction_method = cfg['meshing']['color_mesh_extraction_method']
        self.get_largest_components = cfg['meshing']['get_largest_components']
        self.depth_test = cfg['meshing']['depth_test']
        
        self.bound = slam.bound
        self.nice = slam.nice
        self.verbose = slam.verbose

        self.marching_cubes_bound = torch.from_numpy(
            np.array(cfg['mapping']['marching_cubes_bound']) * self.scale)

        self.frame_reader = get_dataset(cfg, args, self.scale, device='cpu')
        self.n_img = len(self.frame_reader)

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = slam.H, slam.W, slam.fx, slam.fy, slam.cx, slam.cy

    def point_masks(self, input_points, keyframe_dict, estimate_c2w_list,
                    idx, device, get_mask_use_all_frames=False):
        """
        Split the input points into seen, unseen, and forcast,
        according to the estimated camera pose and depth image.

        Args:
            input_points (tensor): input points.
            keyframe_dict (list): list of keyframe info dictionary.
            estimate_c2w_list (tensor): estimated camera pose.
            idx (int): current frame index.
            device (str): device name to compute on.

        Returns:
            seen_mask (tensor): the mask for seen area.
            forecast_mask (tensor): the mask for forecast area.
            unseen_mask (tensor): the mask for unseen area.
        """
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        if not isinstance(input_points, torch.Tensor):
            input_points = torch.from_numpy(input_points)
        input_points = input_points.clone().detach()
        seen_mask_list = []
        forecast_mask_list = []
        unseen_mask_list = []
        for i, pnts in enumerate(
                torch.split(input_points, self.points_batch_size, dim=0)):
            points = pnts.to(device).float()
            # should divide the points into three parts, seen and forecast and unseen
            # seen: union of all the points in the viewing frustum of keyframes
            # forecast: union of all the points in the extended edge of the viewing frustum of keyframes
            # unseen: all the other points

            seen_mask = torch.zeros((points.shape[0])).bool().to(device)
            forecast_mask = torch.zeros((points.shape[0])).bool().to(device)
            if get_mask_use_all_frames:
                for i in range(0, idx + 1, 1):
                    c2w = estimate_c2w_list[i].cpu().numpy()
                    w2c = np.linalg.inv(c2w)
                    w2c = torch.from_numpy(w2c).to(device).float()
                    ones = torch.ones_like(
                        points[:, 0]).reshape(-1, 1).to(device)
                    homo_points = torch.cat([points, ones], dim=1).reshape(
                        -1, 4, 1).to(device).float()  # (N, 4)
                    # (N, 4, 1)=(4,4)*(N, 4, 1)
                    cam_cord_homo = w2c @ homo_points
                    cam_cord = cam_cord_homo[:, :3]  # (N, 3, 1)

                    K = torch.from_numpy(
                        np.array([[fx, .0, cx], [.0, fy, cy],
                                  [.0, .0, 1.0]]).reshape(3, 3)).to(device)
                    cam_cord[:, 0] *= -1
                    uv = K.float() @ cam_cord.float()
                    z = uv[:, -1:] + 1e-8
                    uv = uv[:, :2] / z
                    uv = uv.float()
                    edge = 0
                    cur_mask_seen = (uv[:, 0] < W - edge) & (
                        uv[:, 0] > edge) & (uv[:, 1] < H - edge) & (uv[:, 1] > edge)
                    cur_mask_seen = cur_mask_seen & (z[:, :, 0] < 0)

                    edge = -1000
                    cur_mask_forecast = (uv[:, 0] < W - edge) & (
                        uv[:, 0] > edge) & (uv[:, 1] < H - edge) & (uv[:, 1] > edge)
                    cur_mask_forecast = cur_mask_forecast & (z[:, :, 0] < 0)

                    # forecast
                    cur_mask_forecast = cur_mask_forecast.reshape(-1)
                    # seen
                    cur_mask_seen = cur_mask_seen.reshape(-1)

                    seen_mask |= cur_mask_seen
                    forecast_mask |= cur_mask_forecast
            else:
                for keyframe in keyframe_dict:
                    c2w = keyframe['est_c2w'].cpu().numpy()
                    w2c = np.linalg.inv(c2w)
                    w2c = torch.from_numpy(w2c).to(device).float()
                    ones = torch.ones_like(
                        points[:, 0]).reshape(-1, 1).to(device)
                    homo_points = torch.cat([points, ones], dim=1).reshape(
                        -1, 4, 1).to(device).float()
                    cam_cord_homo = w2c @ homo_points
                    cam_cord = cam_cord_homo[:, :3]

                    K = torch.from_numpy(
                        np.array([[fx, .0, cx], [.0, fy, cy],
                                  [.0, .0, 1.0]]).reshape(3, 3)).to(device)
                    cam_cord[:, 0] *= -1
                    uv = K.float() @ cam_cord.float()
                    z = uv[:, -1:] + 1e-8
                    uv = uv[:, :2] / z
                    uv = uv.float()
                    edge = 0
                    cur_mask_seen = (uv[:, 0] < W - edge) & (
                        uv[:, 0] > edge) & (uv[:, 1] < H - edge) & (uv[:, 1] > edge)
                    cur_mask_seen = cur_mask_seen & (z[:, :, 0] < 0)

                    edge = -1000
                    cur_mask_forecast = (uv[:, 0] < W - edge) & (
                        uv[:, 0] > edge) & (uv[:, 1] < H - edge) & (uv[:, 1] > edge)
                    cur_mask_forecast = cur_mask_forecast & (z[:, :, 0] < 0)

                    if self.depth_test:
                        gt_depth = keyframe['depth'].to(
                            device).reshape(1, 1, H, W)
                        vgrid = uv.reshape(1, 1, -1, 2)
                        # normalized to [-1, 1]
                        vgrid[..., 0] = (vgrid[..., 0] / (W-1) * 2.0 - 1.0)
                        vgrid[..., 1] = (vgrid[..., 1] / (H-1) * 2.0 - 1.0)
                        depth_sample = F.grid_sample(
                            gt_depth, vgrid, padding_mode='zeros', align_corners=True)
                        depth_sample = depth_sample.reshape(-1)
                        max_depth = torch.max(depth_sample)
                        # forecast
                        cur_mask_forecast = cur_mask_forecast.reshape(-1)
                        proj_depth_forecast = -cam_cord[cur_mask_forecast,
                                                        2].reshape(-1)
                        cur_mask_forecast[cur_mask_forecast.clone()] &= proj_depth_forecast < max_depth
                        # seen
                        cur_mask_seen = cur_mask_seen.reshape(-1)
                        proj_depth_seen = - cam_cord[cur_mask_seen, 2].reshape(-1)
                        cur_mask_seen[cur_mask_seen.clone()] &= \
                            (proj_depth_seen < depth_sample[cur_mask_seen]+2.4) \
                            & (depth_sample[cur_mask_seen]-2.4 < proj_depth_seen)
                    else:
                        max_depth = torch.max(keyframe['depth'])*1.1

                        # forecast
                        cur_mask_forecast = cur_mask_forecast.reshape(-1)
                        proj_depth_forecast = -cam_cord[cur_mask_forecast,
                                                        2].reshape(-1)
                        cur_mask_forecast[
                            cur_mask_forecast.clone()] &= proj_depth_forecast < max_depth

                        # seen
                        cur_mask_seen = cur_mask_seen.reshape(-1)
                        proj_depth_seen = - \
                            cam_cord[cur_mask_seen, 2].reshape(-1)
                        cur_mask_seen[cur_mask_seen.clone(
                        )] &= proj_depth_seen < max_depth

                    seen_mask |= cur_mask_seen
                    forecast_mask |= cur_mask_forecast

            forecast_mask &= ~seen_mask
            unseen_mask = ~(seen_mask | forecast_mask)

            seen_mask = seen_mask.cpu().numpy()
            forecast_mask = forecast_mask.cpu().numpy()
            unseen_mask = unseen_mask.cpu().numpy()

            seen_mask_list.append(seen_mask)
            forecast_mask_list.append(forecast_mask)
            unseen_mask_list.append(unseen_mask)

        seen_mask = np.concatenate(seen_mask_list, axis=0)
        forecast_mask = np.concatenate(forecast_mask_list, axis=0)
        unseen_mask = np.concatenate(unseen_mask_list, axis=0)
        return seen_mask, forecast_mask, unseen_mask

    def get_bound_from_frames(self, keyframe_dict, scale=1):
        """
        Get the scene bound (convex hull),
        using sparse estimated camera poses and corresponding depth images.

        Args:
            keyframe_dict (list): list of keyframe info dictionary.
            scale (float): scene scale.

        Returns:
            return_mesh (trimesh.Trimesh): the convex hull.
        """

        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy

        if version.parse(o3d.__version__) >= version.parse('0.13.0'):
            # for new version as provided in environment.yaml
            volume = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=4.0 * scale / 512.0,
                sdf_trunc=0.04 * scale,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
        else:
            # for lower version
            volume = o3d.integration.ScalableTSDFVolume(
                voxel_length=4.0 * scale / 512.0,
                sdf_trunc=0.04 * scale,
                color_type=o3d.integration.TSDFVolumeColorType.RGB8)
        cam_points = []
        for keyframe in keyframe_dict:
            c2w = keyframe['est_c2w'].cpu().numpy()
            # convert to open3d camera pose
            c2w[:3, 1] *= -1.0
            c2w[:3, 2] *= -1.0
            w2c = np.linalg.inv(c2w)
            cam_points.append(c2w[:3, 3])
            depth = keyframe['depth'].cpu().numpy()
            color = keyframe['color'].cpu().numpy()

            depth = o3d.geometry.Image(depth.astype(np.float32))
            color = o3d.geometry.Image(np.array(
                (color * 255).astype(np.uint8)))

            intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color,
                depth,
                depth_scale=1,
                depth_trunc=1000,
                convert_rgb_to_intensity=False)
            volume.integrate(rgbd, intrinsic, w2c)

        cam_points = np.stack(cam_points, axis=0)
        mesh = volume.extract_triangle_mesh()
        mesh_points = np.array(mesh.vertices)
        points = np.concatenate([cam_points, mesh_points], axis=0)
        o3d_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        mesh, _ = o3d_pc.compute_convex_hull()
        mesh.compute_vertex_normals()
        if version.parse(o3d.__version__) >= version.parse('0.13.0'):
            mesh = mesh.scale(self.clean_mesh_bound_scale, mesh.get_center())
        else:
            mesh = mesh.scale(self.clean_mesh_bound_scale, center=True)
        points = np.array(mesh.vertices)
        faces = np.array(mesh.triangles)
        return_mesh = trimesh.Trimesh(vertices=points, faces=faces)
        return return_mesh

    def eval_points(self, p, decoders, c=None, stage='color', device='cuda:0'):
        """
        Evaluates the occupancy and/or color value for the points.

        Args:
            p (tensor, N*3): point coordinates.
            decoders (nn.module decoders): decoders.
            c (dicts, optional): feature grids. Defaults to None.
            stage (str, optional): query stage, corresponds to different levels. Defaults to 'color'.
            device (str, optional): device name to compute on. Defaults to 'cuda:0'.

        Returns:
            ret (tensor): occupancy (and color) value of input points.
        """

        p_split = torch.split(p, self.points_batch_size)
        bound = self.bound
        rets = []
        for pi in p_split:
            # mask for points out of bound
            mask_x = (pi[:, 0] < bound[0][1]) & (pi[:, 0] > bound[0][0])
            mask_y = (pi[:, 1] < bound[1][1]) & (pi[:, 1] > bound[1][0])
            mask_z = (pi[:, 2] < bound[2][1]) & (pi[:, 2] > bound[2][0])
            mask = mask_x & mask_y & mask_z

            pi = pi.unsqueeze(0)
            if self.nice:
                ret = decoders(pi, c_grid=c, stage=stage)
            else:
                ret = decoders(pi, c_grid=None)
            ret = ret.squeeze(0)
            if len(ret.shape) == 1 and ret.shape[0] == 4:
                ret = ret.unsqueeze(0)

            ret[~mask, 3] = 100
            rets.append(ret)

        ret = torch.cat(rets, dim=0)
        return ret

    def get_grid_uniform(self, resolution):
        """
        Get query point coordinates for marching cubes.

        Args:
            resolution (int): marching cubes resolution.

        Returns:
            (dict): points coordinates and sampled coordinates for each axis.
        """
        bound = self.marching_cubes_bound

        padding = 0.05
        x = np.linspace(bound[0][0] - padding, bound[0][1] + padding,
                        resolution)
        y = np.linspace(bound[1][0] - padding, bound[1][1] + padding,
                        resolution)
        z = np.linspace(bound[2][0] - padding, bound[2][1] + padding,
                        resolution)

        xx, yy, zz = np.meshgrid(x, y, z)
        grid_points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
        grid_points = torch.tensor(np.vstack(
            [xx.ravel(), yy.ravel(), zz.ravel()]).T,
            dtype=torch.float)

        return {"grid_points": grid_points, "xyz": [x, y, z]}

    def get_mesh(self,
                 mesh_out_file,
                 c,
                 decoders,
                 keyframe_dict,
                 estimate_c2w_list,
                 idx,
                 device='cuda:0',
                 show_forecast=False,
                 color=True,
                 clean_mesh=True,
                 get_mask_use_all_frames=False):
        """
        Extract mesh from scene representation and save mesh to file.

        Args:
            mesh_out_file (str): output mesh filename.
            c (dicts): feature grids.
            decoders (nn.module): decoders.
            keyframe_dict (list):  list of keyframe info.
            estimate_c2w_list (tensor): estimated camera pose.
            idx (int): current processed camera ID.
            device (str, optional): device name to compute on. Defaults to 'cuda:0'.
            show_forecast (bool, optional): show forecast. Defaults to False.
            color (bool, optional): whether to extract colored mesh. Defaults to True.
            clean_mesh (bool, optional): whether to clean the output mesh 
                                        (remove outliers outside the convexhull and small geometry noise). 
                                        Defaults to True.
            get_mask_use_all_frames (bool, optional): 
                whether to use all frames or just keyframes when getting the seen/unseen mask. Defaults to False.
        """
        with torch.no_grad():

            grid = self.get_grid_uniform(self.resolution)
            points = grid['grid_points']
            points = points.to(device)

            if show_forecast:

                seen_mask, forecast_mask, unseen_mask = self.point_masks(
                    points, keyframe_dict, estimate_c2w_list, idx, device=device, 
                    get_mask_use_all_frames=get_mask_use_all_frames)

                forecast_points = points[forecast_mask]
                seen_points = points[seen_mask]

                z_forecast = []
                for i, pnts in enumerate(
                        torch.split(forecast_points,
                                    self.points_batch_size,
                                    dim=0)):
                    z_forecast.append(
                        self.eval_points(pnts, decoders, c, 'coarse',
                                         device).cpu().numpy()[:, -1])
                z_forecast = np.concatenate(z_forecast, axis=0)
                z_forecast += 0.2

                z_seen = []
                for i, pnts in enumerate(
                        torch.split(seen_points, self.points_batch_size,
                                    dim=0)):
                    z_seen.append(
                        self.eval_points(pnts, decoders, c, 'fine',
                                         device).cpu().numpy()[:, -1])
                z_seen = np.concatenate(z_seen, axis=0)

                z = np.zeros(points.shape[0])
                z[seen_mask] = z_seen
                z[forecast_mask] = z_forecast
                z[unseen_mask] = -100

            else:
                mesh_bound = self.get_bound_from_frames(
                    keyframe_dict, self.scale)
                z = []
                mask = []
                for i, pnts in enumerate(torch.split(points, self.points_batch_size, dim=0)):
                    mask.append(mesh_bound.contains(pnts.cpu().numpy()))
                mask = np.concatenate(mask, axis=0)
                for i, pnts in enumerate(torch.split(points, self.points_batch_size, dim=0)):
                    z.append(self.eval_points(pnts, decoders, c, 'fine',
                                              device).cpu().numpy()[:, -1])

                z = np.concatenate(z, axis=0)
                z[~mask] = 100

            z = z.astype(np.float32)

            try:
                if version.parse(
                        skimage.__version__) > version.parse('0.15.0'):
                    # for new version as provided in environment.yaml
                    verts, faces, normals, values = skimage.measure.marching_cubes(
                        volume=z.reshape(
                            grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                            grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
                        level=self.level_set,
                        spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                                 grid['xyz'][1][2] - grid['xyz'][1][1],
                                 grid['xyz'][2][2] - grid['xyz'][2][1]))
                else:
                    # for lower version
                    verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(
                        volume=z.reshape(
                            grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                            grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
                        level=self.level_set,
                        spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                                 grid['xyz'][1][2] - grid['xyz'][1][1],
                                 grid['xyz'][2][2] - grid['xyz'][2][1]))
            except:
                print(
                    'marching_cubes error. Possibly no surface extracted from the level set.'
                )
                return

            # convert back to world coordinates
            vertices = verts + np.array(
                [grid['xyz'][0][0], grid['xyz'][1][0], grid['xyz'][2][0]])

            if clean_mesh:
                if show_forecast:
                    points = vertices
                    mesh = trimesh.Trimesh(vertices=vertices,
                                           faces=faces,
                                           process=False)
                    mesh_bound = self.get_bound_from_frames(
                        keyframe_dict, self.scale)
                    contain_mask = []
                    for i, pnts in enumerate(
                            np.array_split(points, self.points_batch_size,
                                           axis=0)):
                        contain_mask.append(mesh_bound.contains(pnts))
                    contain_mask = np.concatenate(contain_mask, axis=0)
                    not_contain_mask = ~contain_mask
                    face_mask = not_contain_mask[mesh.faces].all(axis=1)
                    mesh.update_faces(~face_mask)
                else:
                    points = vertices
                    mesh = trimesh.Trimesh(vertices=vertices,
                                           faces=faces,
                                           process=False)
                    seen_mask, forecast_mask, unseen_mask = self.point_masks(
                        points, keyframe_dict, estimate_c2w_list, idx, device=device, 
                        get_mask_use_all_frames=get_mask_use_all_frames)
                    unseen_mask = ~seen_mask
                    face_mask = unseen_mask[mesh.faces].all(axis=1)
                    mesh.update_faces(~face_mask)

                # get connected components
                components = mesh.split(only_watertight=False)
                if self.get_largest_components:
                    areas = np.array([c.area for c in components], dtype=np.float)
                    mesh = components[areas.argmax()]
                else:
                    new_components = []
                    for comp in components:
                        if comp.area > self.remove_small_geometry_threshold * self.scale * self.scale:
                            new_components.append(comp)
                    mesh = trimesh.util.concatenate(new_components)
                vertices = mesh.vertices
                faces = mesh.faces

            if color:
                if self.color_mesh_extraction_method == 'direct_point_query':
                    # color is extracted by passing the coordinates of mesh vertices through the network
                    points = torch.from_numpy(vertices)
                    z = []
                    for i, pnts in enumerate(
                            torch.split(points, self.points_batch_size, dim=0)):
                        z_color = self.eval_points(
                            pnts.to(device).float(), decoders, c, 'color',
                            device).cpu()[..., :3]
                        z.append(z_color)
                    z = torch.cat(z, axis=0)
                    vertex_colors = z.numpy()

                elif self.color_mesh_extraction_method == 'render_ray_along_normal':
                    # for imap*
                    # render out the color of the ray along vertex normal, and assign it to vertex color
                    import open3d as o3d
                    mesh = o3d.geometry.TriangleMesh(
                        vertices=o3d.utility.Vector3dVector(vertices),
                        triangles=o3d.utility.Vector3iVector(faces))
                    mesh.compute_vertex_normals()
                    vertex_normals = np.asarray(mesh.vertex_normals)
                    rays_d = torch.from_numpy(vertex_normals).to(device)
                    sign = -1.0
                    length = 0.1
                    rays_o = torch.from_numpy(
                        vertices+sign*length*vertex_normals).to(device)
                    color_list = []
                    batch_size = self.ray_batch_size
                    gt_depth = torch.zeros(vertices.shape[0]).to(device)
                    gt_depth[:] = length
                    for i in range(0, rays_d.shape[0], batch_size):
                        rays_d_batch = rays_d[i:i+batch_size]
                        rays_o_batch = rays_o[i:i+batch_size]
                        gt_depth_batch = gt_depth[i:i+batch_size]
                        depth, uncertainty, color = self.renderer.render_batch_ray(
                            c, decoders, rays_d_batch, rays_o_batch, device, 
                            stage='color', gt_depth=gt_depth_batch)
                        color_list.append(color)
                    color = torch.cat(color_list, dim=0)
                    vertex_colors = color.cpu().numpy()

                vertex_colors = np.clip(vertex_colors, 0, 1) * 255
                vertex_colors = vertex_colors.astype(np.uint8)

                # cyan color for forecast region
                if show_forecast:
                    seen_mask, forecast_mask, unseen_mask = self.point_masks(
                        vertices, keyframe_dict, estimate_c2w_list, idx, device=device,
                        get_mask_use_all_frames=get_mask_use_all_frames)
                    vertex_colors[forecast_mask, 0] = 0
                    vertex_colors[forecast_mask, 1] = 255
                    vertex_colors[forecast_mask, 2] = 255

            else:
                vertex_colors = None

            vertices /= self.scale
            mesh = trimesh.Trimesh(vertices, faces, vertex_colors=vertex_colors)
            mesh.export(mesh_out_file)
            if self.verbose:
                print('Saved mesh at', mesh_out_file)
                