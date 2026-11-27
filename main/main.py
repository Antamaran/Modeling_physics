import matplotlib.pyplot as plt
import numpy as np

from object import Object


V0 = 0 #veritical velocity.
V1 = 0 # horizontal velocity.
T0 = 0 # initial time.
H0 = 1000 # initial height.
I0 = 10 # initial impulse.
T_end = 100 # end time
step = 1 # step in time
drag_coeff = 2 
air_dens = 1
G = 9.81
triangle_matrix = np.array([[[-1, 0, 0], [1, 0, 0], [0, 1, 0]]])
FALLING_OBJECT = Object(volume=None, surface_area=triangle_matrix, mass=10)

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

def euler_method(H0,I0,V0, T0, T_end, step, M, G, drag_coeff, air_dens,area, height):
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
        velocity = euler_step_velocity(
            step, current_velocity, mass, gravity, drag_coeff, air_dens, area
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
    dv_dt = np.array([
        differntial_velocity(FALLING_OBJECT.mass, G, drag_coeff, air_dens, velocity, area)
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

def main():
    area = FALLING_OBJECT.mesh_area
    tX,tV, tH = euler_method(H0,I0,V0, T0, T_end, step, FALLING_OBJECT.mass, G, drag_coeff, air_dens, area, H0)
    tV_a = analytical_solution(
        np.asarray(tX) - tX[0], FALLING_OBJECT.mass, air_dens, drag_coeff,
        area, -tV[0], G
    )
    error = calculate_error(tV, tV_a)
    print("Velocity RMSE (m/s):", error)
    fig, ax = plt.subplots()
    plot_directional_field(tX, tV, ax)
    ax.plot(tX, tV, color="tab:red", linewidth=2, label="Euler solution")
    ax.plot(tX, tV_a, color="tab:blue", linewidth=2, label="Analytical solution")
    ax.set_xlabel("Time t")
    ax.set_ylabel("Velocity v")
    ax.set_title("Directional field for dv/dt")
    ax.grid(True)
    ax.legend()
    plt.show()



if __name__ == "__main__":
    main()
