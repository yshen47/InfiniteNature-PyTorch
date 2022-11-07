import random

import numpy as np
from .surface_point_cloud import SurfacePointCloud
from .utils import scale_to_unit_cube, scale_to_unit_sphere, get_raster_points, check_voxels
import trimesh
from collections import Counter


def get_surface_point_cloud(mesh, surface_point_method='scan', bounding_radius=None, scan_count=100,
                            scan_resolution=400, sample_point_count=100000, calculate_normals=True):
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("The mesh parameter must be a trimesh mesh.")

    if bounding_radius is None:
        bounding_radius = np.max(np.linalg.norm(mesh.vertices, axis=1)) * 1.1

    if surface_point_method == 'scan':
        return surface_point_cloud.create_from_scans(mesh, bounding_radius=bounding_radius, scan_count=scan_count,
                                                     scan_resolution=scan_resolution,
                                                     calculate_normals=calculate_normals)
    elif surface_point_method == 'sample':
        return surface_point_cloud.sample_from_mesh(mesh, sample_point_count=sample_point_count,
                                                    calculate_normals=calculate_normals)
    else:
        raise ValueError('Unknown surface point sampling method: {:s}'.format(surface_point_method))


def mesh_to_sdf(mesh, query_points, surface_point_method='scan', sign_method='normal', bounding_radius=None,
                scan_count=100, scan_resolution=400, sample_point_count=10000000, normal_sample_count=11):
    if not isinstance(query_points, np.ndarray):
        raise TypeError('query_points must be a numpy array.')
    if len(query_points.shape) != 2 or query_points.shape[1] != 3:
        raise ValueError('query_points must be of shape N ✕ 3.')

    if surface_point_method == 'sample' and sign_method == 'depth':
        print("Incompatible methods for sampling points and determining sign, using sign_method='normal' instead.")
        sign_method = 'normal'

    point_cloud = get_surface_point_cloud(mesh, surface_point_method, bounding_radius, scan_count, scan_resolution,
                                          sample_point_count, calculate_normals=sign_method == 'normal')

    if sign_method == 'normal':
        return point_cloud.get_sdf_in_batches(query_points, use_depth_buffer=False)
    elif sign_method == 'depth':
        return point_cloud.get_sdf_in_batches(query_points, use_depth_buffer=True, sample_count=sample_point_count)
    else:
        raise ValueError('Unknown sign determination method: {:s}'.format(sign_method))


def mesh_to_voxels(input_pcd, input_mesh, sub_world_box, z_rotation, voxel_resolution=64, surface_point_method='scan',
                   sign_method='normal', scan_count=100, scan_resolution=400, sample_point_count=10000000,
                   normal_sample_count=11, pad=False, check_result=False, mesh_length=0, mesh_center=None,
                   return_in_original_scale=True):
    mesh, scale, translation = scale_to_unit_sphere(input_mesh, mesh_length, mesh_center)

    input_pcd_points = np.array(input_pcd.points) - translation
    input_pcd_points /= scale

    if surface_point_method == 'sample' and sign_method == 'depth':
        print("Incompatible methods for sampling points and determining sign, using sign_method='normal' instead.")
        sign_method = 'normal'

    surface_point_cloud = SurfacePointCloud(None,
                                            points=input_pcd_points,
                                            normals=np.array(input_pcd.normals),
                                            scans=None
                                            )
    voxel_sdfs, points = surface_point_cloud.get_voxels(voxel_resolution, z_rotation, sign_method == 'depth',
                                                        normal_sample_count,
                                                        pad, check_result)
    if return_in_original_scale:
        points = points * scale + translation
        voxel_sdfs = voxel_sdfs * scale
    voxel_sdfs = sanity_check_sdfs_with_multi_gurantee_points(sub_world_box, input_mesh, voxel_sdfs, points)
    points = points.reshape((voxel_resolution, voxel_resolution, voxel_resolution, 3))
    voxel_sdfs = voxel_sdfs.reshape((voxel_resolution, voxel_resolution, voxel_resolution))

    # SDFFiltering().marching_cube(voxel_sdfs)

    voxel_sdfs = SDFFiltering().filter_mesh(voxel_sdfs, single_mesh_size_threshold=10, filter_negative_sdf=True)
    voxel_sdfs = SDFFiltering().filter_mesh(voxel_sdfs, single_mesh_size_threshold=10, filter_negative_sdf=False)

    return points, voxel_sdfs


# Sample some uniform points and some normally distributed around the surface as proposed in the DeepSDF paper
def sample_sdf_near_surface(input_pcd, input_mesh, sub_world_box, number_of_points=500000, known_query_points=None,
                            surface_point_method='sample', sign_method='normal',
                            scan_count=100, scan_resolution=400, sample_point_count=1000000, normal_sample_count=11,
                            mesh_length=0, mesh_center=None,
                            min_size=0, return_gradients=False, return_in_original_scale=True):
    mesh, scale, translation = scale_to_unit_sphere(input_mesh, mesh_length, mesh_center)

    input_pcd_points = np.array(input_pcd.points) - translation
    input_pcd_points /= scale

    if known_query_points is not None:
        known_query_points = known_query_points - translation
        known_query_points /= scale
    if surface_point_method == 'sample' and sign_method == 'depth':
        print("Incompatible methods for sampling points and determining sign, using sign_method='normal' instead.")
        sign_method = 'normal'

    # surface_point_cloud = get_surface_point_cloud(mesh, surface_point_method, 1, scan_count, scan_resolution, sample_point_count, calculate_normals=sign_method=='normal' or return_gradients)
    surface_point_cloud = SurfacePointCloud(None,
                                            points=input_pcd_points,
                                            normals=np.array(input_pcd.normals),
                                            scans=None
                                            )
    points, sdfs = surface_point_cloud.sample_sdf_near_surface(known_query_points, number_of_points,
                                                               surface_point_method == 'scan', sign_method,
                                                               normal_sample_count, min_size, return_gradients)

    if return_in_original_scale:
        points = points * scale + translation
        sdfs = sdfs * scale

    sdfs = sanity_check_sdfs_with_multi_gurantee_points(sub_world_box, input_mesh, sdfs, points)

    return points, sdfs


def sanity_check_sdfs_with_multi_gurantee_points(sub_world_box, mesh, sdfs, points, num=2):
    gurantee_points = []
    # start_time = time.time()

    for v in sub_world_box.vertices:
        if v[1] < 0:
            v[1] = -5
            gurantee_points.append(v)
    random.shuffle(gurantee_points)
    sdfs = sdfs.flatten()
    neg_sdf_index = []
    for gurantee_point in gurantee_points[:num]:
        neg_sdf_index += sanity_check_sdfs_with_single_gurantee_point(mesh, points, gurantee_point)
    neg_sdf_index = list(set(neg_sdf_index))
    sdfs = np.abs(sdfs)
    sdfs[neg_sdf_index] = -sdfs[neg_sdf_index]
    # sdfs = sdfs.reshape(sdf_shapes)
    # print("sdf time: ", time.time() - start_time)

    return sdfs


def sanity_check_sdfs_with_single_gurantee_point(mesh, points, gurantee_point):
    # sdf_shapes = sdfs.shape

    ray_manager = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh, False)

    candidates = list(np.arange(len(points)))

    rays_from_gurantee_point_to_intersection_point = points - gurantee_point
    rays_from_gurantee_point_to_intersection_point /= np.linalg.norm(rays_from_gurantee_point_to_intersection_point,
                                                                     axis=1)[:, None]
    _, index_rays, locations = ray_manager.intersects_id(gurantee_point[None,].repeat(len(points), 0),
                                                         rays_from_gurantee_point_to_intersection_point,
                                                         return_locations=True)

    # checked_out_point_inds += list(set(np.arange(len(candidates))) - set(np.unique(index_rays)))

    rays_from_candidate_to_intersection_point = locations - points[index_rays]
    intersect_in_between_flags = np.where(np.matmul(rays_from_gurantee_point_to_intersection_point[index_rays][:, None],
                                                    rays_from_candidate_to_intersection_point[
                                                        ..., None]).squeeze() < 0)[0]
    intersect_in_between_counts = Counter(list(index_rays[intersect_in_between_flags]))

    odd_ind = []
    for (k, v) in intersect_in_between_counts.items():
        if v % 2 == 1:
            odd_ind.append(k)
    neg_sdf_index = list(np.array(candidates)[np.array(odd_ind)])
    return neg_sdf_index


class SDFFiltering(object):

    def make_water(self, i, j, k, name):
        frontier = [(i, j, k)]
        curr_count = 0
        while True:
            new_frontier = []
            for node in frontier:
                i, j, k = node
                if not(i < 0 or j < 0 or k < 0 or i >= self.voxel.shape[0] or j >= self.voxel.shape[1] or k >= self.voxel.shape[2] \
                        or (self.voxel[i][j][k] >= 0 if self.filter_negative_sdf else self.voxel[i][j][k] <= 0)):
                    if self.history[i][j][k] == 0:
                        self.history[i][j][k] = name
                        curr_count += 1
                        new_frontier.append((i + 1, j, k))
                        new_frontier.append((i - 1, j, k))
                        new_frontier.append((i, j + 1, k))
                        new_frontier.append((i, j - 1, k))
                        new_frontier.append((i, j, k + 1))
                        new_frontier.append((i, j, k - 1))
            if len(new_frontier) == 0:
                break
            frontier = new_frontier
        return curr_count

    def filter_mesh(self, voxel, single_mesh_size_threshold, filter_negative_sdf=True):
        self.filter_negative_sdf = filter_negative_sdf
        if len(voxel) == 0:
            return 0
        self.voxel = voxel
        self.history = np.zeros_like(voxel)
        n, m, o = voxel.shape
        name = 1
        count_map = {}
        invalid_names = []

        for i in range(n):
            for j in range(m):
                for k in range(o):
                    if (voxel[i][j][k] < 0 if self.filter_negative_sdf else voxel[i][j][k] > 0) and self.history[i][j][k] == 0:
                        name += 1
                        c = self.make_water(i, j, k, name)
                        count_map[name] = c
                        if c < single_mesh_size_threshold:
                            invalid_names.append(name)
        for invalid_name in invalid_names:
            voxel[self.history == invalid_name] = -voxel[self.history == invalid_name]

        # self.marching_cube(voxel)
        return voxel

    def marching_cube(self, voxels):
        import skimage
        vertices, faces, normals, _ = skimage.measure.marching_cubes(voxels, level=0)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
        mesh.show()