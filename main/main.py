import matplotlib.pyplot as plt

V0 = 0
T0 = 0
T_end = 100
step = 0.0001
drag_coeff = 1
air_dens = 1
G = 9.81
M = 10


def equation(mass, G, drag_coeff, air_dens, velocity):
    DvDt = G - (1/2 * drag_coeff * air_dens * velocity**2) / mass
    
    return DvDt

def euler_step(step, time, V_current, M, G, drag_coeff, air_dens):
    time = time + step
    DvDt = equation(M, G, drag_coeff, air_dens, V_current)
    V_current = V_current + DvDt * step
    
    return time, V_current

def euler_method(V0, T0, T_end, step, M, G, drag_coeff, air_dens):
    
    time_coordinates = []
    velocity_coordinates = []
    
    velocity = V0
    time = T0
    
    
    while time < T_end:
        
        time, velocity = euler_step(step, time, velocity, M, G, drag_coeff, air_dens)
        
        time_coordinates.append(time)
        velocity_coordinates.append(velocity)
    
    return time_coordinates, velocity_coordinates

def plot(x,y):
    fig, ax = plt.subplots()
    ax.plot(x,y)
    plt.show()


def main():
    tX,tY = euler_method(V0, T0, T_end, step, M, G, drag_coeff, air_dens)
    plot(tX, tY)



if __name__ == "__main__":
    main()
