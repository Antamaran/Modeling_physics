import main as simulation


class World:
    def __init__(self):
        self.bodies = []

    def step(self, dt=simulation.step):
        """Advance each body's vertical position and velocity by one Euler step."""
        for body in self.bodies:
            velocity = body.vertical_velocity
            density = simulation.air_density_height(body.z)
            body.vertical_velocity = simulation.euler_step_velocity(
                dt, velocity, body.mass, simulation.G, simulation.drag_coeff,
                density, body.mesh_area
            )
            body.z = simulation.euler_step_height(dt, velocity, body.z)

    def add_body(self,body):
        self.bodies.append(body)
        
        
