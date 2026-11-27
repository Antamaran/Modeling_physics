import numpy as np


class Object:
    def __init__(self, volume, surface_area, mass, x=0, y=0, z=0, 
                 x_velocity=0, y_velocity=0,
                 z_velocity=0,x_acceleration=0,
                 y_acceleration=0,z_acceleration=0, *, center_of_mass=(0, 0, 0),
                 principal_moments=None, angular_velocity=(0, 0, 0)):
        self.volume = volume
        self.mass = mass
        self.x = x
        self.y = y
        self.z = z

        self.x_velocity = x_velocity
        self.y_velocity = y_velocity 
        self.z_velocity = z_velocity

        self.x_acceleration=x_acceleration
        self.y_acceleration=y_acceleration
        self.z_acceleration=z_acceleration
        # Center of mass in mesh coordinates (m), measured from the mesh origin.
        self.center_of_mass = np.array(center_of_mass, dtype=float)
        # Moments of inertia about the three body axes (kg m²). Larger values resist spin changes.
        self.principal_moments = (
            None if principal_moments is None else np.array(principal_moments, dtype=float)
        )
        self.angular_velocity = np.array(angular_velocity, dtype=float)  # world axes, rad/s
        
        self.orientation = np.eye(3) # identity matrix

        self.surface_area = np.asarray(surface_area, dtype=float)

        if self.surface_area.ndim != 3 or self.surface_area.shape[1:] != (3, 3): # if not one tirangle or not a mesh of traingles, raise error
            raise ValueError("surface_area must have shape (N, 3, 3)")

    @property
    def mesh_area(self):
        """Return the total area of the triangular faces."""
        vertices = self.surface_area
        first_edges = vertices[:, 1] - vertices[:, 0] #AB of all triangles
        second_edges = vertices[:, 2] - vertices[:, 0] #AC of all triangles
        cross_products = np.cross(first_edges, second_edges) # |AB*AC|
        return float(0.5 * np.linalg.norm(cross_products, axis=1).sum()) #||AB*AC||*0.5 sum. 
    
    @property
    def apply_force(self, force): # theta is measured in respect to y=0. 
        force=force/self.mass
        self.x_acceleration=force[0]
        self.y_acceleration=force[1]
        self.z_acceleration=force[2]
