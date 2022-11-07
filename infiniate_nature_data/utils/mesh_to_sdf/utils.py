import trimesh
import numpy as np


def scale_to_unit_sphere(mesh, mesh_length, mesh_center):
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()

    vertices = mesh.vertices - mesh_center
    distances = np.linalg.norm(vertices, axis=1)
    vertices /= mesh_length
    scale = mesh_length
    translation = mesh_center
    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces), scale, translation


def scale_to_unit_cube(mesh):
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()

    vertices = mesh.vertices - mesh.bounding_box.centroid
    vertices *= 2 / np.max(mesh.bounding_box.extents)

    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)

def get_raster_points(voxel_resolution, z_rotation):
    points = np.meshgrid(
        np.linspace(-0.5, 0.5, voxel_resolution),
        np.linspace(-0.8, 0.2, voxel_resolution),
        np.linspace(-0.5, 0.5, voxel_resolution)
    )
    points = np.stack(points)
    points = np.swapaxes(points, 1, 2)
    points = points.reshape(3, -1).transpose().astype(np.float32)
    points = (z_rotation @ points.T).T
    return points


def check_voxels(voxels):
    block = voxels[:-1, :-1, :-1]
    d1 = (block - voxels[1:, :-1, :-1]).reshape(-1)
    d2 = (block - voxels[:-1, 1:, :-1]).reshape(-1)
    d3 = (block - voxels[:-1, :-1, 1:]).reshape(-1)

    max_distance = max(np.max(d1), np.max(d2), np.max(d3))
    return max_distance < 2.0 / voxels.shape[0] * 3 ** 0.5 * 1.1


def sample_uniform_points_in_unit_sphere(amount):
    unit_sphere_points = np.random.uniform(-np.sqrt(2)/2, np.sqrt(2)/2, size=(amount * 2 + 20, 3))
    unit_sphere_points = unit_sphere_points[np.linalg.norm(unit_sphere_points, axis=1) < np.sqrt(2)/2]

    points_available = unit_sphere_points.shape[0]
    if points_available < amount:
        # This is a fallback for the rare case that too few points are inside the unit sphere
        result = np.zeros((amount, 3))
        result[:points_available, :] = unit_sphere_points
        result[points_available:, :] = sample_uniform_points_in_unit_sphere(amount - points_available)
        return result
    else:
        return unit_sphere_points[:amount, :]
