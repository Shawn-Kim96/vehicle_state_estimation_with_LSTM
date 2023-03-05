import numpy as np
import pandas as pd

from src.data.six_dof_dyanmics.rigidbody_6dof import *

params = {
    "mass": 1.0,  # [kg]
    "inertia": np.eye(3),
    "gain for quaternion normalization": 1.0,
    "time interval": 0.001,  # [s]
    "init_LLA": [37.4849885, 127.0358544, 5],  # [deg], [deg], [m]
    "init_time": "2020-12-01 01:23:45",  # '%Y-%m-%d %H:%M:%S'
    "gravity_constant": 9.81,  # [m/s^2]
}
params["inverse of inertia"] = np.linalg.inv(params["inertia"])

sim_result = SimData()
sim = RigidBody6DOF(sim_result, params)


def force_and_moment_example(t, x):
    F = np.random.uniform(-1.0, 1.0, 3)
    # M = np.random.uniform(-1.0, 1.0, 3)
    # F = np.array([0., 0., 0.])
    M = np.array([0.0, 0.0, 0.0])

    # if 1.0 <= t < 2.0:
    #     F[1] = 1.
    # else:
    #     F[1] = 0.
    #
    # if 7.0 <= t < 8.0:
    #     M[0] = 1.
    # else:
    #     M[0] = 0.
    return F, M


sim.run_sim(
    initial_state=np.hstack(
        (
            [0.0, 0.0, 0.0],  # position
            [0.0, 0.0, 0.0],  # velocity
            [0.0, 0.0, 0.0],  # angular velocity
            euler2quat([0.1, 0.2, 0.3]),  # quaternion (from euler angle )
        )
    ),
    timespan=[0, 10],
    maneuver_func=force_and_moment_example,
)

# import matplotlib.pyplot as plt
#
# plt.figure()
#
# plt.subplot(6,1,1)
# plt.plot(sim_result.F)
# plt.ylabel('F')
#
# plt.subplot(6,1,2)
# plt.plot(sim_result.M)
# plt.ylabel('M')
#
# plt.subplot(6,1,3)
# plt.plot(sim_result.x[:, 0:3])
# plt.ylabel('r')
#
# plt.subplot(6,1,4)
# plt.plot(sim_result.x[:, 3:6])
# plt.ylabel('v')
#
# plt.subplot(6,1,5)
# plt.plot(sim_result.x[:, 6:9])
# plt.ylabel('w')
#
# plt.subplot(6,1,6)
# plt.plot(sim_result.euler)
# plt.ylabel('euler')
#
#
# plt.show()


def save_simresult_as_df(sim_result, save_path):
    df = pd.DataFrame(
        {
            "Fx": sim_result.F[:, 0],
            "Fy": sim_result.F[:, 1],
            "Fz": sim_result.F[:, 2],
            "Mx": sim_result.M[:, 0],
            "My": sim_result.M[:, 1],
            "Mz": sim_result.M[:, 2],
            "rx": sim_result.x[:, 0],
            "ry": sim_result.x[:, 1],
            "rz": sim_result.x[:, 2],
            "vx": sim_result.x[:, 3],
            "vy": sim_result.x[:, 4],
            "vz": sim_result.x[:, 5],
            "wx": sim_result.x[:, 6],
            "wy": sim_result.x[:, 7],
            "wz": sim_result.x[:, 8],
            "e1": sim_result.euler[:, 0],
            "e2": sim_result.euler[:, 1],
            "e3": sim_result.euler[:, 2],
            "dvx": sim_result.x_dot[:, 3],
            "dvy": sim_result.x_dot[:, 4],
            "dvz": sim_result.x_dot[:, 5],
        }
    )
    df.to_csv(save_path)


if __name__ == "__main__":
    save_simresult_as_df(sim_result, "6dof_F_uniform_input.csv")
