import matplotlib.pyplot as plt
import numpy as np
from object import Object

INITIAL_X_VELOCITY = 0  # m/s
INITIAL_Y_VELOCITY = 0  # m/s
INITIAL_Z_VELOCITY = 0  # m/s, positive upward
T0 = 0 # initial time.
H0 = 1000 # initial height.
I0 = 10 # initial impulse.
T_end = 100 # end time
step = 0.0001 # step in time
SPIN_TIME_STEP = 0.01  # seconds between updates of position, velocity and spin
drag_coeff = 2 
air_dens = 1
G = 9.81
triangle_matrix = np.array([[[-1, 0, 0], [1, 0, 0], [0, 1, 0]]])
# A uniform 6 kg triangular plate with 4 kg of ballast at the mesh origin.
# The ballast shifts the center of mass away from the plate centroid, giving drag a lever arm.
FALLING_OBJECT = Object(
    volume=None, surface_area=triangle_matrix, mass=10, x=0, y=0, z=H0,
    x_velocity=INITIAL_X_VELOCITY, y_velocity=INITIAL_Y_VELOCITY,
    z_velocity=INITIAL_Z_VELOCITY, center_of_mass=(0, 0.2, 0),
    principal_moments=(0.6, 1.0, 1.6)  # kg m² about the center of mass
)
P0=1.225 #kg/m^3, air density at sea water level.


def air_density_height(height, sea_level_density=P0): #height in meters.
    #ρ(h)=ρ0e^(−h/H)
    return sea_level_density * np.exp(-height / 8500)

def analytical_solution(t, m, rho, Cd, A, V0, g=9.81):
    """Return vertical velocity at elapsed time t under gravity and quadratic drag.

    V0 is the initial downward-positive velocity. The result uses an upward-positive
    coordinate, so falling velocities are negative. t may be a scalar or array.
    """
    t = np.asarray(t)
    k = (rho * Cd * A) / (2 * m)
    if k == 0:
        return -V0 - g * t

    terminal_speed = np.sqrt(g / k)
    rate = np.sqrt(g * k)

    if V0 < 0:
        initial_angle = np.arctan(-V0 / terminal_speed)
        apex_time = initial_angle / rate
        ascent_time = np.minimum(t, apex_time)
        descent_time = np.maximum(t - apex_time, 0)
        return terminal_speed * (
            np.tan(rate * (apex_time - ascent_time)) - np.tanh(rate * descent_time)
        )

    speed_ratio = V0 / terminal_speed
    drag_term = np.tanh(rate * t)
    return -terminal_speed * (speed_ratio + drag_term) / (1 + speed_ratio * drag_term)

def air_resistance(air_dens, velocity,drag_coeff,area):
    return 0.5 * air_dens * velocity**2 * drag_coeff * area

def differntial_velocity(mass, G, drag_coeff, air_dens, velocity, area):
    drag = air_resistance(air_dens, velocity, drag_coeff, area)

    if velocity > 0:
        drag = -drag

    return drag / mass - G

def differntial_hieght(velocity):
    return velocity

def euler_step_velocity(step, V_current, M, G, drag_coeff, air_dens, area):
    DvDt = differntial_velocity(M, G, drag_coeff, air_dens, V_current, area)
    V_current = V_current + DvDt * step
    
    return V_current

def euler_step_time(step, time):
    time = time + step

    return time

def euler_step_height(step,V_current, height):
    DhDt=differntial_hieght(V_current)
    current_height=height+DhDt*step
    return current_height

def euler_method(H0, I0, V0, T0, T_end, step, M, G, air_dens, area, height,
                 *, drag_coeff=drag_coeff):
    initial_velocity = V0 + I0 / M
    return _integrate_trajectory(
        H0, initial_velocity, T0, T_end, step, M, G, drag_coeff, air_dens,
        area, height
    )


def _integrate_trajectory(initial_height, current_velocity, time, end_time, step,
                          mass, gravity, drag_coeff, air_dens, area, height):
    time_coordinates = [time]
    velocity_coordinates = [current_velocity]
    height_coordinates = [initial_height]

    while time < end_time and height >= 0:
        local_density = air_density_height(height, air_dens)
        velocity = euler_step_velocity(
            step, current_velocity, mass, gravity, drag_coeff, local_density, area
        )
        height = euler_step_height(step, current_velocity, height)
        time = euler_step_time(step, time)
        current_velocity = velocity

        time_coordinates.append(time)
        velocity_coordinates.append(current_velocity)
        height_coordinates.append(height)

    return time_coordinates, velocity_coordinates, height_coordinates

def plot(x,y):
    fig, ax = plt.subplots()
    ax.plot(x,y)
    plt.show()

def plot_directional_field(t_values, velocity_values, ax):
    # Plots directional field. 
    t_min, t_max = min(t_values), max(t_values)
    v_min, v_max = min(velocity_values), max(velocity_values)
    v_padding = max(0.1 * (v_max - v_min), 1)

    t_grid, v_grid = np.meshgrid(
        np.linspace(t_min, t_max, 93),
        np.linspace(v_min - v_padding, v_max + v_padding, 31)
    )

    area = FALLING_OBJECT.mesh_area
    effective_gravity = G - FALLING_OBJECT.z_acceleration
    dv_dt = np.array([
        differntial_velocity(
            FALLING_OBJECT.mass, effective_gravity, drag_coeff, air_dens, velocity, area
        )
        for velocity in v_grid.ravel()
    ]).reshape(v_grid.shape)

    magnitude = np.sqrt(1 + dv_dt**2)
    t_direction = 1 / magnitude
    v_direction = dv_dt / magnitude

    ax.quiver(
        t_grid, v_grid, t_direction, v_direction,
        angles="xy", scale_units="xy", scale=6,
        color="tab:blue", alpha=0.85, width=0.0015,
        headwidth=2.5, headlength=3, headaxislength=2.5
    )

def calculate_error(tV, tV_a):
    return np.sqrt(np.mean((np.asarray(tV) - tV_a) ** 2))

def simulate_object(body, impulse, start_time, end_time, time_step):
    """Return times, vertical velocities and positions, and update the body."""
    times, velocities, positions = simulate_trajectory(
        body, impulse, start_time, end_time, time_step
    )
    return times, velocities[2], positions


def simulate_trajectory(body, impulse, start_time, end_time, time_step):
    """Return times and (x, y, z) histories of velocity and position.

    Apply an initial upward impulse in N·s. Horizontal accelerations are
    constant. Vertical motion also includes gravity and quadratic air drag.
    Update the body to the final recorded state.
    """
    if time_step <= 0:
        raise ValueError("time_step must be positive")

    times, vertical_velocities, heights = euler_method(
        body.z, impulse, body.z_velocity, start_time, end_time, time_step,
        body.mass, G - body.z_acceleration, P0, body.mesh_area, body.z,
        drag_coeff=drag_coeff
    )
    elapsed_time = np.asarray(times) - times[0]
    half_time_squared = 0.5 * elapsed_time**2
    x_positions = body.x + body.x_velocity * elapsed_time + body.x_acceleration * half_time_squared
    y_positions = body.y + body.y_velocity * elapsed_time + body.y_acceleration * half_time_squared
    x_velocities = body.x_velocity + body.x_acceleration * elapsed_time
    y_velocities = body.y_velocity + body.y_acceleration * elapsed_time

    body.x, body.y, body.z = float(x_positions[-1]), float(y_positions[-1]), heights[-1]
    body.x_velocity = float(x_velocities[-1])
    body.y_velocity = float(y_velocities[-1])
    body.z_velocity = vertical_velocities[-1]
    return times, (x_velocities, y_velocities, vertical_velocities), (x_positions, y_positions, heights)


def rotation_increment(angular_velocity, dt):
    """Build a world-frame rotation from angular velocity using Rodrigues' formula."""
    # The rotation vector points along the spin axis and has length |omega| * dt in radians.
    x, y, z = angular_velocity * dt
    # This matrix takes the cross product of the rotation vector with any vector it multiplies.
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    angle = np.linalg.norm(angular_velocity) * dt
    # The sinc form keeps the coefficients defined even when the spin is zero.
    return np.eye(3) + np.sinc(angle / np.pi) * skew + (
        0.5 * np.sinc(angle / (2 * np.pi)) ** 2 * (skew @ skew)
    )


def simulate_spinning_fall(body, impulse, start_time, end_time, time_step):
    """Return time, linear velocity, position and angular velocity histories."""
    if not np.isfinite(time_step) or time_step <= 0:
        raise ValueError("time_step must be finite and positive")
    moments = body.principal_moments
    if moments is None or moments.shape != (3,) or not np.all(np.isfinite(moments) & (moments > 0)):
        raise ValueError("principal_moments must contain three positive finite values")

    # An area vector points along the face normal and has length equal to the face area.
    triangles = body.surface_area
    area_vectors = 0.5 * np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]) # surface area.


    centroids = triangles.mean(axis=1, keepdims=True) # mean of each triangle.
    sample_position=triangle_matrix+centroids/2 # distance halfway between each vertex. So point in which drag acts
    sample_offsets = (sample_position - body.center_of_mass).reshape(-1, 3) # -com, to transform com to origin. 
    sample_area_vectors = np.repeat(area_vectors / 3, 3, axis=0)

    # Track the center of mass and apply the initial upward impulse as a velocity change.
    position = np.array([body.x, body.y, body.z], dtype=float)
    velocity = np.array([body.x_velocity, body.y_velocity, body.z_velocity + impulse / body.mass])
    acceleration = np.array([body.x_acceleration, body.y_acceleration, body.z_acceleration - G])
    orientation = body.orientation
    omega = body.angular_velocity
    # Rotate omega into the body axes, apply L = I * omega, then return L in world axes.
    angular_momentum = orientation @ (moments * (orientation.T @ omega))

    time = start_time
    times = [time]
    positions = [position]
    velocities = [velocity]
    spins = [omega]
    while time < end_time and position[2] >= 0:
        # Shorten the last step at the time limit or when the center of mass reaches the ground.
        dt = min(time_step, end_time - time)
        landing = velocity[2] < 0 and position[2] <= -velocity[2] * dt
        if landing:
            dt = position[2] / -velocity[2]
            if dt == 0:
                break

        # Rotate the row vectors for the sample positions and face normals into world axes.
        offsets = sample_offsets @ orientation.T
        world_area_vectors = sample_area_vectors @ orientation.T
        # Each sample moves with the center of mass plus the local spin velocity.
        local_velocity = velocity + np.cross(omega, offsets)
        # The absolute dot product gives projected area times speed for either side of the face.
        projected_area_times_speed = np.abs(np.sum(world_area_vectors * local_velocity, axis=1))
        density = air_density_height(position[2])
        # Quadratic drag opposes local motion. The offset r and force F give torque r × F.
        forces = -0.5 * density * drag_coeff * projected_area_times_speed[:, None] * local_velocity
        torque = np.cross(offsets, forces).sum(axis=0)

        position = position + dt * velocity
        velocity = velocity + dt * (acceleration + forces.sum(axis=0) / body.mass)
        # Torque is the rate of change of angular momentum. Uniform gravity acts through
        # the center of mass, so only aerodynamic forces contribute torque here.
        angular_momentum = angular_momentum + dt * torque
        # Estimate the average spin using the old orientation, then recalculate omega after rotating.
        next_omega = orientation @ ((orientation.T @ angular_momentum) / moments)
        orientation = rotation_increment(0.5 * (omega + next_omega), dt) @ orientation
        omega = orientation @ ((orientation.T @ angular_momentum) / moments)

        time += dt
        if landing:
            position[2] = 0
        times.append(time)
        positions.append(position)
        velocities.append(velocity)
        spins.append(omega)
        if landing:
            break

    body.x, body.y, body.z = position
    body.x_velocity, body.y_velocity, body.z_velocity = velocity
    body.orientation = orientation
    body.angular_velocity = omega
    return np.asarray(times), np.asarray(velocities), np.asarray(positions), np.asarray(spins)

def plot_position(time_values, coordinates):
    plot_components(
        time_values, coordinates, "Position over time", ("x (m)", "y (m)", "z height (m)")
    )


def plot_components(time_values, components, title, labels):
    elapsed_time = np.asarray(time_values) - time_values[0]
    _, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 7), layout="constrained")
    axes[0].set_title(title)
    for axis, values, label in zip(axes, components, labels):
        axis.plot(elapsed_time, values)
        axis.set_ylabel(label)
        axis.grid(True)
    axes[-1].set_xlabel("Elapsed time (s)")


def plot_vertical_fall():
    times, velocities, coordinates = simulate_trajectory(FALLING_OBJECT, I0, T0, T_end, step)
    vertical_velocities = velocities[2]
    effective_gravity = G - FALLING_OBJECT.z_acceleration
    fig, ax = plt.subplots()
    plot_directional_field(times, vertical_velocities, ax)
    ax.plot(times, vertical_velocities, color="tab:red", linewidth=2, label="Euler solution")
    if effective_gravity > 0:
        analytical_velocities = analytical_solution(
            np.asarray(times) - times[0], FALLING_OBJECT.mass, air_dens, drag_coeff,
            FALLING_OBJECT.mesh_area, -vertical_velocities[0], effective_gravity
        )
        print("Vertical velocity RMSE (m/s):", calculate_error(vertical_velocities, analytical_velocities))
        ax.plot(times, analytical_velocities, color="tab:blue", linewidth=2, label="Analytical solution")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Vertical velocity (m/s)")
    ax.set_title("Vertical velocity directional field")
    ax.grid(True)
    ax.legend()
    plot_position(times, coordinates)
    plot_components(
        times, velocities, "Velocity over time", ("vx (m/s)", "vy (m/s)", "vz (m/s)")
    )


def main():
    times, velocities, positions, angular_velocities = simulate_spinning_fall(
        FALLING_OBJECT, I0, T0, T_end, SPIN_TIME_STEP
    )
    # The history arrays store one row per time. Transpose them to plot one series per axis.
    plot_position(times, positions.T)
    plot_components(
        times, velocities.T, "Velocity over time", ("vx (m/s)", "vy (m/s)", "vz (m/s)")
    )
    plot_components(
        times, angular_velocities.T, "Angular velocity over time",
        ("ωx (rad/s)", "ωy (rad/s)", "ωz (rad/s)")
    )
    plt.show()



if __name__ == "__main__":
    main()
