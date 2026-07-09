# utils.py
# Ray generation and ray sampling utilities

import numpy as np


def normalize(v):
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(norm, 1e-8)


def generate_ray_points(origin, direction, near, far, num_samples):
    """
    Generate 3D sample points along one camera ray.

    Ray equation:

        p(t) = origin + t * direction

    Args:
        origin: ray origin, shape (3,)
        direction: ray direction, shape (3,)
        near: starting distance along the ray
        far: ending distance along the ray
        num_samples: number of samples along the ray

    Returns:
        points: sampled 3D points, shape (num_samples, 3)
        delta: distance between consecutive samples
    """

    origin = np.array(origin, dtype=np.float32)
    direction = np.array(direction, dtype=np.float32)
    direction = direction / np.maximum(np.linalg.norm(direction), 1e-8)

    delta = (far - near) / num_samples

    # Use midpoint sampling
    t_values = near + (np.arange(num_samples) + 0.5) * delta

    points = origin[None, :] + t_values[:, None] * direction[None, :]

    return points, delta


def generate_camera_rays(width, height, fov_degrees, camera_position):
    """
    Generate one ray per pixel.

    Args:
        width: image width
        height: image height
        fov_degrees: field of view in degrees
        camera_position: camera position, shape (3,)

    Returns:
        origins: shape (height, width, 3)
        directions: shape (height, width, 3)
    """

    camera_position = np.array(camera_position, dtype=np.float32)

    aspect = width / height
    fov = np.deg2rad(fov_degrees)

    screen_height = 2.0 * np.tan(fov / 2.0)
    screen_width = aspect * screen_height

    xs = np.linspace(-screen_width / 2.0, screen_width / 2.0, width)
    ys = np.linspace(screen_height / 2.0, -screen_height / 2.0, height)

    xv, yv = np.meshgrid(xs, ys)

    # Camera is at z = -3 and looks toward positive z
    directions = np.stack(
        [xv, yv, np.ones_like(xv)],
        axis=-1
    )

    directions = normalize(directions)

    origins = np.broadcast_to(camera_position, directions.shape)

    return origins, directions