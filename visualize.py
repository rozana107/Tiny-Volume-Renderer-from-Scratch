# visualize.py
# Visualize the volumetric scene in 3D

import numpy as np
import matplotlib.pyplot as plt


def visualize_density_3d(
    scene,
    grid_size=60,
    bounds=(-1.5, 1.5),
    density_threshold=0.8,
    max_points=20000
):
    """
    Visualize volumetric objects by sampling the 3D density field.

    This does not render the volume like a camera.
    Instead, it shows 3D points where the density is high enough.

    Args:
        scene: Scene object with scene.evaluate(points)
        grid_size: number of samples along x, y, z
        bounds: min and max coordinate of the 3D grid
        density_threshold: only show points with density above this value
        max_points: maximum number of points to draw
    """

    low, high = bounds

    xs = np.linspace(low, high, grid_size)
    ys = np.linspace(low, high, grid_size)
    zs = np.linspace(low, high, grid_size)

    x, y, z = np.meshgrid(xs, ys, zs, indexing="ij")

    points = np.stack([x, y, z], axis=-1)
    points_flat = points.reshape(-1, 3)

    sigmas, colors = scene.evaluate(points_flat)

    mask = sigmas > density_threshold

    visible_points = points_flat[mask]
    visible_colors = colors[mask]
    visible_sigmas = sigmas[mask]

    if len(visible_points) == 0:
        print("No points found above the density threshold.")
        return

    # Reduce number of displayed points if there are too many
    if len(visible_points) > max_points:
        indices = np.random.choice(
            len(visible_points),
            size=max_points,
            replace=False
        )

        visible_points = visible_points[indices]
        visible_colors = visible_colors[indices]
        visible_sigmas = visible_sigmas[indices]

    # Convert density to alpha
    alpha = visible_sigmas / np.max(visible_sigmas)
    alpha = np.clip(alpha, 0.05, 0.8)

    rgba_colors = np.concatenate(
        [visible_colors, alpha[:, None]],
        axis=-1
    )

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        visible_points[:, 0],
        visible_points[:, 1],
        visible_points[:, 2],
        c=rgba_colors,
        s=5
    )

    ax.set_title("3D Visualization of Volumetric Spheres")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_zlim(low, high)

    ax.set_box_aspect([1, 1, 1])

    plt.show()