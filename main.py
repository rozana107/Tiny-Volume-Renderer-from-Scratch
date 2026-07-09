import numpy as np
import matplotlib.pyplot as plt
from render import render_image
from scene import create_two_sphere_scene

def test_render_image():
    width = 400
    height = 400
    fov_degrees = 60
    camera_position = np.array([0.0, 0.0, -3.0])
    near = 1.0
    far = 5.0
    num_samples = 128

    scene = create_two_sphere_scene()

    try:
        scene_image = render_image(
            scene,
            width,
            height,
            fov_degrees,
            camera_position,
            near,
            far,
            num_samples
        )
        print("render_image ran successfully. Output shape:", scene_image.shape)
        plt.imshow(scene_image)
        plt.axis("off")
        plt.show()
        plt.imsave("tiny_volume_renderer.png", scene_image)
    except Exception as e:
        print("render_image failed with exception:", e)

if __name__ == "__main__":
    test_render_image()