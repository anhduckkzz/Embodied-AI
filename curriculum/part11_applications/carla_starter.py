"""CARLA starter: connect, spawn a car with a camera, autopilot, save frames.

This is reference code that runs on a machine with the CARLA simulator server
running and the `carla` Python client installed. It is not runnable in this
curriculum's plain sandbox (CARLA needs a GPU server), so it explains clearly if
CARLA is absent.

Run (with the CARLA server already started):
    python curriculum/part11_applications/carla_starter.py

It demonstrates the full client loop: connect, spawn a vehicle, attach an RGB
camera, enable autopilot, collect a few frames, then clean up the actors.
"""
from __future__ import annotations

import os


def main(num_frames: int = 30, out_dir: str = "carla_frames"):
    try:
        import carla
    except ImportError:
        print("The 'carla' client is not installed, or no CARLA server is running.")
        print("Install CARLA (server binary + `pip install carla`) and start the")
        print("server, then run this script. See carla_guide.md.")
        return

    import random

    # 1. Connect to the running CARLA server.
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    actors = []
    try:
        # 2. Spawn a vehicle at a random valid spawn point.
        vehicle_bp = random.choice(blueprint_library.filter("vehicle.*"))
        spawn = random.choice(world.get_map().get_spawn_points())
        vehicle = world.spawn_actor(vehicle_bp, spawn)
        actors.append(vehicle)
        print(f"spawned {vehicle.type_id}")

        # 3. Attach an RGB camera to the vehicle.
        cam_bp = blueprint_library.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", "640")
        cam_bp.set_attribute("image_size_y", "480")
        cam_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle)
        actors.append(camera)

        os.makedirs(out_dir, exist_ok=True)
        saved = {"n": 0}

        def on_image(image):
            if saved["n"] < num_frames:
                image.save_to_disk(os.path.join(out_dir, f"{image.frame:06d}.png"))
                saved["n"] += 1

        camera.listen(on_image)

        # 4. Let the built-in autopilot drive (Traffic Manager).
        vehicle.set_autopilot(True)
        print(f"driving on autopilot, saving {num_frames} frames to {out_dir}/ ...")

        # 5. Tick until enough frames are collected.
        while saved["n"] < num_frames:
            world.wait_for_tick()
        print("done. Replace autopilot with your own controller (Part 14) next.")
    finally:
        for actor in actors:
            actor.destroy()   # always clean up spawned actors


if __name__ == "__main__":
    main()
