from control_utils import CarModel
import numpy as np
import matplotlib.pyplot as plt

# =================================================
#       Front wheel and steering wheel angles
# =================================================
GEM = CarModel(
    carLength = 1.75,
    steerSpeed = 2.5*35/630,
    approx_steerTau = 5.0,
    carAccel = 2.0,
    carDecel = -5.0,
    carDamp = 2.0/11.1,
    steerLimits = (-np.pi*35/180, np.pi*35/180),
    throttleLimits = (0.2, 0.5),
    throttleRateLimits = (-1.0, .25),
    brakeLimits = (0.0, 1.0),
    brakeRateLimits = (-5.0, 5.0)
)

delta1 = np.linspace(-1, 1, 100, endpoint=True)
steer1 = np.array([0.0]*len(delta1))
for i in range(len(delta1)):
    steer1[i] = GEM.delta2steer(delta1[i])

steer2 = np.linspace(-20, 20, 41, endpoint=True)
delta2 = np.array([0.0]*len(steer2))
for i in range(len(steer2)):
    delta2[i] = GEM.steer2delta(steer2[i])

plt.plot(delta1, steer1, label='delta2steer', c='black', lw=1)
plt.plot(delta2, steer2, label='steer2delta', c='r', ls='', marker='.', lw=1)
plt.xlabel('delta')
plt.ylabel('steering wheel')
plt.grid()
plt.legend()
plt.show()

