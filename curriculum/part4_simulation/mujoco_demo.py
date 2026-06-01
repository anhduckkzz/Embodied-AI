"""A minimal, runnable MuJoCo demo: simulate and control a 1-joint pendulum.

Install:  pip install mujoco
Run:      python curriculum/part4_simulation/mujoco_demo.py

This shows the universal simulation loop on a *real* physics engine:
  1. define a robot in MJCF (XML),
  2. step the dynamics,
  3. read sensors (joint angle/velocity),
  4. write an actuator command from a PID controller (reused from robotics/).

It runs headless (no display needed). If you have a screen, set VIEWER=1 to open
the interactive viewer. We reuse the SAME PID you met in Part 9 / robotics.control
so you can see classical control driving a high-fidelity simulator.
"""
from __future__ import annotations

import os
import sys

# Make the repo root importable so `robotics` is found when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# A pendulum: a single hinge joint with a motor, swinging under gravity.
MJCF = """
<mujoco model="pendulum">
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <light pos="0 0 3"/>
    <geom type="plane" size="2 2 0.1" rgba="0.8 0.9 0.8 1"/>
    <body name="pole" pos="0 0 1">
      <joint name="hinge" type="hinge" axis="0 1 0" damping="0.1"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.6" size="0.025" rgba="0.2 0.4 0.9 1"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="torque" joint="hinge" gear="1" ctrlrange="-20 20"/>
  </actuator>
  <sensor>
    <jointpos name="angle" joint="hinge"/>
    <jointvel name="vel" joint="hinge"/>
  </sensor>
</mujoco>
"""


def main():
    try:
        import mujoco
        import numpy as np
    except ImportError:
        print("MuJoCo is not installed. Install it with:\n    pip install mujoco")
        print("\n(This script is written so you can run it on your own machine.)")
        return

    from robotics.control import PID

    model = mujoco.MjModel.from_xml_string(MJCF)
    data = mujoco.MjData(model)

    # Goal: swing the pendulum up and hold it horizontal (angle = pi/2).
    target = np.pi / 2
    pid = PID(kp=30.0, ki=2.0, kd=6.0, setpoint=target, output_limits=(-20, 20))

    print("Controlling a MuJoCo pendulum to angle = pi/2 ...")
    print(f"{'t (s)':>6} {'angle':>8} {'target':>8} {'torque':>8}")
    dt = model.opt.timestep
    for step in range(2000):
        angle = float(data.qpos[0])            # read sensor (joint position)
        data.ctrl[0] = pid(angle, dt)          # write actuator command
        mujoco.mj_step(model, data)            # step the physics
        if step % 200 == 0:
            print(f"{step * dt:6.2f} {angle:8.3f} {target:8.3f} {data.ctrl[0]:8.3f}")

    print(f"\nFinal angle: {float(data.qpos[0]):.3f} rad (target {target:.3f}). "
          "Classical PID + a real physics engine — that's robotics.")

    if os.environ.get("VIEWER") == "1":
        import mujoco.viewer
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                data.ctrl[0] = pid(float(data.qpos[0]), dt)
                mujoco.mj_step(model, data)
                viewer.sync()


if __name__ == "__main__":
    main()
