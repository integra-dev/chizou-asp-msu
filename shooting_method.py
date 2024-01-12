"""
Решается задача классического вариационного исчисления,
в которой на всей оптимальной траектории управление особое

I(x(*)) = Integral|(0,1) f_0(t,x,u)dt
f_0(t,x,u) = (x(t) - 1)^2

dx(t)/dt - u(t) = 0, для любого t из [0,1]
u(t) из R (-inf, inf) для t из [0,1]

x(0) - 1 = 0
x(1) - 1 = 0
"""


import numpy as np
from gekko import GEKKO
import matplotlib.pyplot as plt


# define infinity value
INF = 99999

# Initialize gekko model
m = GEKKO()
# Number of phases
n_phase = 1
t0, t1, steps = 0.0, 1.0, 101
delta_t = (t1-t0) / float(steps)

# Solver options

# m.options.SEQUENTIAL = 1
m.options.SOLVER = 3
m.options.IMODE = 6         # IMODE = 9 corresponds to shooting method, 6 - iterative methods
m.options.NODES = 1         # number of collocation points in the span of each time segment
m.options.OTOL = 1e-6       # epsilon
m.options.RTOL = 1e-6       # epsilon
m.options.MAX_ITER = 1001   # max number of iterations


# Time horizon (for all phases)
m.time = np.linspace(t0, t1, steps)

# Input (constant in IMODE 4)
u = [m.MV(value=1.0, lb=-INF, ub=INF, fixed_initial=True) for i in range(n_phase)]
for ui in u:
    ui.STATUS = 1

# State Variable
x = [m.SV() for i in range(n_phase)]

for xi in x:
    m.fix(xi, pos=0, val=1.0)
    m.fix(xi, pos=len(m.time)-1, val=1.0)

# Equations (different for each phase)
for i in range(n_phase):
    m.Equation(x[i].dt() - u[i] == 0)

# Connect phases together at endpoints
# for i in range(n-1):
#     m.Connection(x[i+1], x[i], 1, len(m.time)-1, 1, nodes)
#     m.Connection(x[i+1], 'calculated',pos1=1,node1=1)

# Objective
J = m.integral((x[0]+1)**2)
m.Minimize(J)

# Solve
m.solve()

# Plot
plt.figure()
for i in range(n_phase):
    plt.title(f'График x, u, J (метод стрельбы). min(J) = {m.options.OBJFCNVAL}\n$t\\in$[0,1], $u_i\\in$$R^{n_phase}$')
    plt.plot(m.time, x[i].value, lw=2, label=rf'$x_{i+1}$ (состояние)')
    plt.plot(m.time, u[i].value, '--', lw=2, label=rf'$u_{i+1}$ (фаз.)')
    plt.plot(m.time, J.value, lw=2, label=rf'$J$ (функц.)')
    plt.grid(True, which='both')
    plt.legend(loc='best')
    plt.xlabel('Время')
    plt.ylabel('Значение')
    plt.tight_layout()
plt.show()
