import matplotlib.pyplot as plt

V0 = 0
T0 = 0
H0 = 0 # initial height.
I0 = 10 # initial impulse.
AREA=1
T_end = 100
step = 0.0001
drag_coeff = 1
air_dens = 1
G = 9.81
M = 10

def air_resistance(air_dens, velocity,drag_coeff,area):# more advanced air resistance.
    return 0.5 * air_dens * velocity**2 *drag_coeff* area

def differntial_velocity(mass, G, drag_coeff, air_dens, velocity, area):
    drag = air_resistance(air_dens, velocity, drag_coeff, area)

    if velocity > 0:
        drag = -drag

    return drag / mass - G

def differntial_hieght(velocity):
    return velocity

def euler_step_velocity(step, time, V_current, M, G, drag_coeff, air_dens, area):
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
    
    
    while time < T_end:

        velocity = euler_step_velocity(step, time, current_velocity, M, G, drag_coeff, air_dens, area)
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


def main():
    tX,tV, tH = euler_method(H0,I0,V0, T0, T_end, step, M, G, drag_coeff, air_dens, AREA, H0)
    plot(tX, tV)
    plot(tX, tH)



if __name__ == "__main__":
    main()
