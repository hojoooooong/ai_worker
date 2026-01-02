from vuer import Vuer, VuerSession
from vuer.schemas import Bodies, DefaultScene
from asyncio import sleep
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Vuer with proper configuration (similar to visualize_vr.py and controller.py)
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


@app.add_handler("BODY_TRACKING_MOVE")
async def on_body_move(event, session):
    """
    Handle incoming BODY_TRACKING_MOVE events from the client.
    event.value should be a BodiesData dictionary:
      { jointName: { matrix: Float32Array-like, ... }, ... }
    """
    try:
        logger.debug(f"BODY_TRACKING_MOVE: key={event.key} ts={getattr(event, 'ts', None)}")

        # Example: print only the first joint to avoid large output
        if event.value:
            first_joint, first_data = next(iter(event.value.items()))
            logger.debug(
                f"{first_joint}, matrix_len={len(first_data.get('matrix', [])) if first_data else None}"
            )
    except Exception as e:
        logger.error(f"Error processing body tracking movement: {e}")
        import traceback
        traceback.print_exc()


@app.spawn(start=True)
async def main(session: VuerSession):
    """
    Add the Bodies element to the scene and start streaming body tracking data.
    """
    logger.info("Body tracking main scene started")

    # Set default scene with VR enabled
    session.set @ DefaultScene(
        frameloop="always",
        handTracking=False,  # Using body tracking instead
    )

    # Add the Bodies element to the scene and start streaming body tracking data
    session.upsert(
        Bodies(
            key="body_tracking",  # Optional unique identifier (default: "body_tracking")
            stream=True,  # Must be True to start streaming data
            fps=30,  # Send data at 30 frames per second
            hideIndicate=False,  # Hide joint indicators in the scene but still stream data
            showFrame=False,  # Display coordinate frames at each joint
            frameScale=0.02,  # Scale of the coordinate frames or markers
        ),
        to="children",
    )

    logger.info("Body tracking enabled and streaming")

    # Keep the session alive
    while True:
        await sleep(1)
