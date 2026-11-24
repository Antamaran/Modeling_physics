import numpy as np


class Object:
    def __init__(self, volume, surface_area, mass, x=0, y=0, z=0, vertical_velocity=0):
        self.volume = volume
        self.mass = mass
        self.x = x
        self.y = y
        self.z = z
        self.vertical_velocity = vertical_velocity
        self.surface_area = np.asarray(surface_area, dtype=float)
        if self.surface_area.ndim != 3 or self.surface_area.shape[1:] != (3, 3): # if not one tirangle or not a mesh of traingles, raise error
            raise ValueError("surface_area must have shape (N, 3, 3)")

    @property
    def mesh_area(self):
        """Return the total area of the triangular faces."""
        vertices = self.surface_area
        first_edges = vertices[:, 1] - vertices[:, 0] #AB
        second_edges = vertices[:, 2] - vertices[:, 0] #AC
        cross_products = np.cross(first_edges, second_edges) # gives areaof triangle as vector. 
        return float(0.5 * np.linalg.norm(cross_products, axis=1).sum())

