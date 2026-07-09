# main.py

import matplotlib.pyplot as plt

from scene import create_two_sphere_scene
from render import render_image, render_image_fast
from visualize import visualize_density_3d


def main():
    scene = create_two_sphere_scene()

    width = 400
    height = 400
    fov_degrees = scene.camera.fov_degrees
    camera_position = scene.camera.position

    near = 1.0
    far = 5.0
    num_samples = 128

    image = render_image_fast(
        scene,
        width,
        height,
        fov_degrees,
        camera_position,
        near,
        far,
        num_samples,
        row_chunk=32
    )

    plt.imshow(image)
    plt.axis("off")
    plt.show()

    plt.imsave("two_spheres_volume_render.png", image)

    # 3D visualization of the density field
    visualize_density_3d(
        scene,
        grid_size=60,
        bounds=(-1.5, 1.5),
        density_threshold=0.8
    )
    # save the 3D visualization to a file
    plt.savefig("two_spheres_density_3d.png")   

if __name__ == "__main__":
    main()