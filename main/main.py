import matplotlib.pyplot as plt
import numpy as np

V0 = 0 #veritical velocity.
V1 = 0 # horizontal velocity.
T0 = 0
H0 = 1000 # initial height.
I0 = 10 # initial impulse.
AREA = 1
T_end = 100
step = 0.0001
drag_coeff = 2
air_dens = 1
G = 9.81
M = 10

def analytical_solution(t, m, rho, Cd, A, V0, g=9.81):
    k = (rho * Cd * A) / (2 * m)

    D = (np.sqrt(g) + np.sqrt(k) * V0) / (np.sqrt(g) - np.sqrt(k) * V0)

    v = np.sqrt(g / k) * (
        (D * np.exp(2 * np.sqrt(g * k) * t) - 1) /
        (D * np.exp(2 * np.sqrt(g * k) * t) + 1)
    )

    return -v

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
    
    time_coordinates = []
    velocity_coordinates = []
    height_coordinates = []
    
    current_velocity = V0+(I0/M)
    time = T0

    time_coordinates.append(T0)
    velocity_coordinates.append(current_velocity)
    height_coordinates.append(H0)
    
    
    while (time < T_end and height>=0):

        velocity = euler_step_velocity(step, current_velocity, M, G, drag_coeff, air_dens, area)
        height =euler_step_height(step,current_velocity,height)
        time=euler_step_time(step,time)

        current_velocity=velocity
        
        time_coordinates.append(time)
        velocity_coordinates.append(current_velocity)
        height_coordinates.append(height)
    
    return time_coordinates, velocity_coordinates,height_coordinates

def plot(x,y):
    fig, ax = plt.subplots()
    ax.plot(x,y)
    plt.show()

def plot_directional_field(t_values, velocity_values, ax):
    t_min, t_max = min(t_values), max(t_values)
    v_min, v_max = min(velocity_values), max(velocity_values)
    v_padding = max(0.1 * (v_max - v_min), 1)

    t_grid, v_grid = np.meshgrid(
        np.linspace(t_min, t_max, 93),
        np.linspace(v_min - v_padding, v_max + v_padding, 31)
    )

    dv_dt = np.array([
        differntial_velocity(M, G, drag_coeff, air_dens, velocity, AREA)
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
    return np.sqrt(np.sum((tV-tV_a)**2))/len(tV)

def main():
    tX,tV, tH = euler_method(H0,I0,V0, T0, T_end, step, M, G, drag_coeff, air_dens, AREA, H0)
    tV_a = np.vectorize(analytical_solution, excluded="t")(np.linspace(T0, T_end, len(tV)), M, air_dens, drag_coeff, AREA, V0, G)
    error = calculate_error(tV, tV_a)
    print("The error is :", error)
    fig, ax = plt.subplots()
    plot_directional_field(tX, tV, ax)
    ax.plot(tX, tV, color="tab:red", linewidth=2, label="Euler solution")
    ax.plot(np.linspace(T0, T_end, len(tV)), tV_a, color="tab:blue", linewidth=2, label="Analytical solution")
    ax.set_xlabel("Time t")
    ax.set_ylabel("Velocity v")
    ax.set_title("Directional field for dv/dt")
    ax.grid(True)
    ax.legend()
    plt.show()



if __name__ == "__main__":
    main()
