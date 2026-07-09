# scene.py
# Simple 3D scene with 2 soft volumetric spheres.
# The spheres are red and blue and are positioned in the scene.

import numpy as np


class Camera:
    """
    Simple camera class.
    For now, this only stores camera information.
    Ray generation will be handled in the renderer file.
    """

    def __init__(
        self,
        position=np.array([0.0, 0.0, -3.0]),
        look_at=np.array([0.0, 0.0, 0.0]),
        up=np.array([0.0, 1.0, 0.0]),
        fov_degrees=60.0
    ):
        self.position = np.array(position, dtype=np.float32)
        self.look_at = np.array(look_at, dtype=np.float32)
        self.up = np.array(up, dtype=np.float32)
        self.fov_degrees = fov_degrees


class VolumeSphere:
    """
    A soft volumetric sphere.

    Instead of a hard surface, the sphere has a density field:

        sigma(x) = sigma_max * exp(-d^2 / (2 * radius^2))

    where d is the distance from the sphere center.
    """

    def __init__(self, center, radius, color, sigma_max=10.0):
        self.center = np.array(center, dtype=np.float32)
        self.radius = float(radius)
        self.color = np.array(color, dtype=np.float32)
        self.sigma_max = float(sigma_max)

    def density_fn(self, points):
        """
        Compute density at one or many 3D points.

        Input:
            points: shape (3,) or (..., 3)

        Output:
            density: scalar or array with shape (...)
        """

        points = np.array(points, dtype=np.float32)

        diff = points - self.center # difference between the point and the sphere center
        dist2 = np.sum(diff * diff, axis=-1) 

        sigma = self.sigma_max * np.exp( 
            -dist2 / (2.0 * self.radius * self.radius)
        )

        return sigma

    def color_fn(self, points): # return the color of the sphere at the given points
        """
        Return the color of the sphere at the given points.

        For now, the sphere has a constant color.
        """

        points = np.array(points, dtype=np.float32)

        if points.ndim == 1: # if the point is a single point, return the color of the sphere
            return self.color 

        shape = points.shape[:-1] + (3,) # if the point is a batch of points, return the color of the sphere for each point
        return np.broadcast_to(self.color, shape) 


class Scene:
    """
    Scene containing volumetric objects.

    The renderer should query this scene using:

        sigma = scene.density_fn(point)
        color = scene.color_fn(point)

    or:

        sigma, color = scene.evaluate(point)
    """

    def __init__(self):
        self.objects = []
        self.camera = Camera()
        self.background_color = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def add_object(self, obj):
        self.objects.append(obj)

    def set_camera(self, camera):
        self.camera = camera

    def set_background_color(self, color):
        self.background_color = np.array(color, dtype=np.float32)

    def density_fn(self, points):
        """
        Total density of all volumetric objects at a point.

        If multiple spheres overlap, their densities are added.
        """

        if len(self.objects) == 0:
            points = np.array(points, dtype=np.float32)
            return np.zeros(points.shape[:-1], dtype=np.float32)

        total_density = 0.0

        for obj in self.objects:
            total_density += obj.density_fn(points)

        return total_density

    def color_fn(self, points):
        """
        Compute color at a point.

        Since multiple volumetric objects may overlap, we use
        density-weighted color blending:

            c = sum(sigma_k * color_k) / sum(sigma_k)

        This means the denser sphere contributes more color.
        """

        points = np.array(points, dtype=np.float32)

        total_density = self.density_fn(points)

        if points.ndim == 1:
            weighted_color = np.zeros(3, dtype=np.float32)

            for obj in self.objects:
                sigma = obj.density_fn(points)
                color = obj.color_fn(points)
                weighted_color += sigma * color

            if total_density > 1e-8:
                return weighted_color / total_density
            else:
                return self.background_color

        else:
            weighted_color = np.zeros(points.shape, dtype=np.float32)

            for obj in self.objects:
                sigma = obj.density_fn(points)[..., None]
                color = obj.color_fn(points)
                weighted_color += sigma * color

            total_density_safe = np.maximum(total_density[..., None], 1e-8)

            return weighted_color / total_density_safe

    def evaluate(self, points):
        """
        Return both density and color.

        This is the most convenient function to call inside the renderer.
        """

        sigma = self.density_fn(points)
        color = self.color_fn(points)

        return sigma, color


def create_two_sphere_scene():
    """
    Create a default scene with two soft volumetric spheres.
    """

    scene = Scene()

    red_sphere = VolumeSphere(
        center=[-0.45, 0.0, 0.0],
        radius=0.45,
        color=[1.0, 0.1, 0.1],
        sigma_max=10.0
    )

    blue_sphere = VolumeSphere(
        center=[0.55, 0.0, 0.0],
        radius=0.35,
        color=[0.1, 0.2, 1.0],
        sigma_max=10.0
    )

    scene.add_object(red_sphere)
    scene.add_object(blue_sphere)

    return scene


# if __name__ == "__main__":
#     scene = create_two_sphere_scene()

#     test_point = np.array([0.0, 0.0, 0.0])

#     sigma, color = scene.evaluate(test_point)

#     print("Density at test point:", sigma)
#     print("Color at test point:", color)

