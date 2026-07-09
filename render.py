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




