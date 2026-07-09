# render.py
# Render rays through the volumetric scene

import numpy as np
from utils import generate_ray_points, generate_camera_rays


def volume_render_ray(scene, origin, direction, near, far, num_samples):
    """
    Perform volume rendering along a single ray.
    """

    points, delta = generate_ray_points(
        origin,
        direction,
        near,
        far,
        num_samples
    )

    sigmas, colors = scene.evaluate(points)

    rgb = np.zeros(3, dtype=np.float32)
    T = 1.0

    for i in range(num_samples):
        sigma = sigmas[i]
        color = colors[i]

        alpha = 1.0 - np.exp(-sigma * delta)

        weight = T * alpha
        rgb += weight * color

        T *= 1.0 - alpha

        if T < 1e-4:
            break

    opacity = 1.0 - T

    return rgb, opacity


# it runs slow
def render_image(scene, width, height, fov_degrees, camera_position, near, far, num_samples):
    """
    Render a full image by shooting one ray per pixel.
    """

    origins, directions = generate_camera_rays(
        width,
        height,
        fov_degrees,
        camera_position
    )

    image = np.zeros((height, width, 3), dtype=np.float32)

    for y in range(height):
        for x in range(width):
            rgb, opacity = volume_render_ray(
                scene,
                origins[y, x],
                directions[y, x],
                near,
                far,
                num_samples
            )

            image[y, x] = rgb

    return np.clip(image, 0.0, 1.0)

# it runs fast
def render_image_fast(scene, width, height, fov_degrees, camera_position, near, far, num_samples, row_chunk=32):
    """
    Faster volume renderer using NumPy vectorization.

    Instead of rendering one pixel at a time, this renders several rows at once.
    This avoids slow Python loops over every pixel.
    """

    origins, directions = generate_camera_rays(
        width,
        height,
        fov_degrees,
        camera_position
    )

    image = np.zeros((height, width, 3), dtype=np.float32)

    delta = (far - near) / num_samples

    t_values = near + (np.arange(num_samples, dtype=np.float32) + 0.5) * delta

    for y0 in range(0, height, row_chunk):
        y1 = min(y0 + row_chunk, height)

        origins_chunk = origins[y0:y1]       # shape: (chunk, W, 3)
        directions_chunk = directions[y0:y1] # shape: (chunk, W, 3)

        # points shape: (chunk, W, num_samples, 3)
        points = (
            origins_chunk[:, :, None, :]
            + t_values[None, None, :, None] * directions_chunk[:, :, None, :]
        )

        sigmas, colors = scene.evaluate(points)

        # sigmas shape: (chunk, W, num_samples)
        # colors shape: (chunk, W, num_samples, 3)

        alpha = 1.0 - np.exp(-sigmas * delta)

        # Compute front-to-back transmittance:
        # T_i = product_{j<i} (1 - alpha_j)
        one_minus_alpha = 1.0 - alpha

        T = np.cumprod(one_minus_alpha + 1e-10, axis=2)

        # Make T exclusive instead of inclusive.
        # T_exclusive at sample 0 should be 1.
        T_exclusive = np.concatenate(
            [
                np.ones_like(T[:, :, :1]),
                T[:, :, :-1]
            ],
            axis=2
        )

        weights = T_exclusive * alpha

        rgb = np.sum(weights[..., None] * colors, axis=2)

        image[y0:y1] = rgb

    return np.clip(image, 0.0, 1.0)


