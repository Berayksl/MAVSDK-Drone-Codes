#SAMPLE SCRIPT FOR WAYPOINT FOLLOWING USING MAVSDK
#!/usr/bin/env python3

import asyncio
from mavsdk import System
from mavsdk.offboard import (OffboardError, PositionNedYaw)


async def run():

    drone = System()
    print('here')
    await drone.connect(system_address="udp://:4550")

    status_text_task = asyncio.ensure_future(print_status_text(drone))

    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print(f"-- Connected to drone!")
            break
    
    # print("Waiting for drone to have a global position estimate...")
    # async for health in drone.telemetry.health():
    #     if health.is_global_position_ok and health.is_home_position_ok:
    #         print("-- Global position estimate OK")
    #         break
    
    print("-- Arming")
    await drone.action.arm()
    
    print("-- Taking off")
    await drone.action.takeoff()
    await asyncio.sleep(5)

    # await drone.offboard.set_position_ned(
    # PositionNedYaw(0.0, 0.0, -1, 90.0))
    # await asyncio.sleep(5)

    await drone.offboard.set_position_ned(PositionNedYaw(-0.6, 0.0, -1, 90.0)) #order: y,x,z 
    #+x goes forward, +y goes left (towards to optitrack computer)
    await asyncio.sleep(3)
    await drone.offboard.set_position_ned(
    PositionNedYaw(1.5, 1.50, -1, 90.0))
    await asyncio.sleep(5)
    await drone.offboard.set_position_ned(
    PositionNedYaw(1.5, 0, -1, 90.0))
    await asyncio.sleep(5)
    await drone.offboard.set_position_ned(
    PositionNedYaw(0, 0, -1, 90.0))
    await asyncio.sleep(5)
    print("-- Landing")
    await drone.action.land()
    status_text_task.cancel()


async def print_status_text(drone):
    try:
        async for status_text in drone.telemetry.status_text():
            print(f"Status: {status_text.type}: {status_text.text}")
    except asyncio.CancelledError:
        return


if __name__ == "__main__":
    # Run the asyncio loop
    asyncio.run(run())