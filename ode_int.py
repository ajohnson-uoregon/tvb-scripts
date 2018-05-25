import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt

exec("def fun(t, y):\n\ttheta, omega,  = y\n\tdydt = [omega, -0.25*omega-5.0*np.sin(theta), ]\n\treturn dydt\n")

def fun1(t, y):
    theta, omega = y
    dydt = [omega,
            -0.25*omega - 5*np.sin(theta)]
    return dydt

y0 = [np.pi-0.1, 0.0]
t = np.linspace(0, 10, 101)

sol = scipy.integrate.solve_ivp(fun, [0,10], y0, t_eval=t)
#sol = scipy.integrate.odeint(fun1, y0, t)
#print(sol)

plt.plot(t, sol.y[0], 'b', label='theta(t)')
plt.plot(t, sol.y[1], 'g', label='omega(t)')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.show()
