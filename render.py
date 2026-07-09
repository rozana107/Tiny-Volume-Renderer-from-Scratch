# volume_render_ray(origin, direction, near, far, num_samples)
# render a single ray through the scene
# return the color of the ray 

import numpy as np
from utils import generate_ray_points, generate_camera_rays

def volume_render_ray(scene, origin, direction, near, far, num_samples):
    """
    Perform volume rendering along a single ray.

    Args:
        scene (Scene): Scene object supporting .evaluate(points)
        origin (np.ndarray): Ray origin (shape (3,))
        direction (np.ndarray): Ray direction (shape (3,))
        near (float): Near bound
        far (float): Far bound
        num_samples (int): Number of samples along the ray

    Returns:
        rgb (np.ndarray): Accumulated color (shape (3,))
        opacity (float): Opacity along the ray
    """
    delta = (far - near) / num_samples

    samples, points = generate_ray_points(near, far, num_samples, origin, direction)
    sigmas, colors = scene.evaluate(points)  # sigmas: (N,), colors: (N,3)
    rgb = np.zeros(3, dtype=np.float32)
    T = 1.0

    for i in range(num_samples):
        sigma = sigmas[i]  # density at this sample
        color = colors[i]  # color at this sample, (3,)
        alpha = 1.0 - np.exp(-sigma * delta)  # probability of light termination here

        weight = T * alpha     # how much this point contributes
        rgb += weight * color  # add its color weighted by prob of reaching and terminating

        T *= (1.0 - alpha)     # update transmission for next step

        # Optional early termination if essentially opaque
        if T < 1e-4:
            break

    opacity = 1.0 - T
    return rgb, opacity


def render_image(width, height, fov_degrees, camera_position, near, far, num_samples):
    origins, directions = generate_camera_rays(
        width,
        height,
        fov_degrees,
        camera_position
    )

    image = np.zeros((height, width, 3))

    for y in range(height):
        for x in range(width):
            rgb, opacity = volume_render_ray(
                origins[y, x],
                directions[y, x],
                near,
                far,
                num_samples
            )
            image[y, x] = rgb

    return np.clip(image, 0.0, 1.0)

