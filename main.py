# main.py

import matplotlib.pyplot as plt

from scene import create_two_sphere_scene
from render import render_image


def main():
    scene = create_two_sphere_scene()

    width = 400
    height = 400
    fov_degrees = scene.camera.fov_degrees
    camera_position = scene.camera.position

    near = 1.0
    far = 5.0
    num_samples = 128

    image = render_image(
        scene,
        width,
        height,
        fov_degrees,
        camera_position,
        near,
        far,
        num_samples
    )

    plt.imshow(image)
    plt.axis("off")
    plt.show()

    plt.imsave("two_spheres_volume_render.png", image)


if __name__ == "__main__":
    main()