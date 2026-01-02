from vuer import Vuer, VuerSession
from vuer.schemas import MotionControllers, DefaultScene
from asyncio import sleep
from datetime import datetime
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Vuer with proper configuration (similar to visualize_vr.py)
app = Vuer(
    host='0.0.0.0',  # Listen on all network interfaces
    port=8012,
    queries=dict(
        grid=False,
        xr=True,
        vr=True,
        ws=f"wss://192.168.50.203"  # WebSocket connection URL
    ),
    queue_len=10,
    cert="./cert.pem",
    key="./key.pem",
    reconnect=True,
    max_retries=10,
    retry_delay=1.0
)

prev_bstate = None

@app.add_handler("CONTROLLER_MOVE")
async def handler(
    event,
    session: VuerSession,
):
    global prev_bstate
    try:
        if event and event.value and "leftState" in event.value:
            bstate = event.value["leftState"]["triggerValue"]
            
            # Format and print controller data in a readable way
            print("\n" + "="*60)
            print(f"Event Key: {event.key}")
            print(f"Timestamp: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
            print("-"*60)
            
            # Print left controller matrix (4x4)
            if "left" in event.value:
                left_matrix = np.array(event.value["left"]).reshape(4, 4)
                print("\n[Left Controller Matrix]")
                print(left_matrix)
            
            # Print left controller state
            if "leftState" in event.value:
                left_state = event.value["leftState"]
                print("\n[Left Controller State]")
                print(f"  Trigger: {left_state.get('trigger', False)} (Value: {left_state.get('triggerValue', 0):.3f})")
                print(f"  Squeeze: {left_state.get('squeeze', False)} (Value: {left_state.get('squeezeValue', 0):.3f})")
                print(f"  Touchpad: {left_state.get('touchpad', False)} (Value: {left_state.get('touchpadValue', [0, 0])})")
                print(f"  Thumbstick: {left_state.get('thumbstick', False)} (Value: {left_state.get('thumbstickValue', [0, 0])})")
                print(f"  A Button: {left_state.get('aButton', False)}")
                print(f"  B Button: {left_state.get('bButton', False)}")
            
            # Print right controller matrix (4x4)
            if "right" in event.value:
                right_matrix = np.array(event.value["right"]).reshape(4, 4)
                print("\n[Right Controller Matrix]")
                print(right_matrix)
            
            # Print right controller state
            if "rightState" in event.value:
                right_state = event.value["rightState"]
                print("\n[Right Controller State]")
                print(f"  Trigger: {right_state.get('trigger', False)} (Value: {right_state.get('triggerValue', 0):.3f})")
                print(f"  Squeeze: {right_state.get('squeeze', False)} (Value: {right_state.get('squeezeValue', 0):.3f})")
                print(f"  Touchpad: {right_state.get('touchpad', False)} (Value: {right_state.get('touchpadValue', [0, 0])})")
                print(f"  Thumbstick: {right_state.get('thumbstick', False)} (Value: {right_state.get('thumbstickValue', [0, 0])})")
                print(f"  A Button: {right_state.get('aButton', False)}")
                print(f"  B Button: {right_state.get('bButton', False)}")
            
            print("="*60)

            if prev_bstate != bstate and bstate:
                # Pulse the gamepad according to the trigger value
                session.upsert @ MotionControllers(
                    key="motion-controller",
                    left=True,
                    right=True,
                    pulseLeftStrength=bstate,
                    pulseLeftDuration=100,
                    pulseLeftHash=f"{datetime.now()}:{bstate:.3f}",
                )
                logger.debug(f"Controller pulse triggered: {bstate}")

            prev_bstate = bstate
    except Exception as e:
        logger.error(f"Error processing controller movement: {e}")
        import traceback
        traceback.print_exc()

@app.spawn(start=True)
async def main(session: VuerSession):
    logger.info("Controller main scene started")
    
    # Set default scene with VR enabled
    session.set @ DefaultScene(
        frameloop="always",
        handTracking=False,  # Using motion controllers instead
    )
    
    # Important: You need to set the `stream` option to `True` to start
    # streaming the controller movement.
    session.upsert @ MotionControllers(
        stream=True,
        key="motion-controller",
        left=True,
        right=True
    )
    
    logger.info("Motion controllers enabled and streaming")

    while True:
        await sleep(1)
