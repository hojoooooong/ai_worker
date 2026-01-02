from vuer import Vuer
from vuer.schemas import Hands, DefaultScene
import numpy as np
import asyncio
from multiprocessing import Array, Queue, Event
import time
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VRVisualizer:
    def __init__(self):
        logger.info("Initializing VR Visualizer...")

        # Initialize VR data arrays
        self.left_hand_shared = Array('d', 16)
        self.right_hand_shared = Array('d', 16)
        self.left_landmarks_shared = Array('d', 75)
        self.right_landmarks_shared = Array('d', 75)
        self.head_matrix_shared = Array('d', 16)

        logger.info("Shared arrays initialized")

        # Initialize Vuer for VR connection
        logger.info("Setting up Vuer connection...")
        self.vuer = Vuer(
            host='0.0.0.0',  # Listen on all network interfaces
            port=8012,
            queries=dict(
                grid=False,
                xr=True,
                vr=True,
                ws=f"wss://192.168.0.8:8012"  # WebSocket connection URL
            ),
            queue_len=10,
            cert="./cert.pem",
            key="./key.pem",
            reconnect=True,
            max_retries=10,
            retry_delay=1.0
        )

        # Add event handlers
        logger.info("Registering event handlers...")
        @self.vuer.add_handler("HAND_MOVE")
        async def on_hand_move_wrapper(event, session):
            logger.debug("Handler called: HAND_MOVE")
            await self.on_hand_move(event, session)

        @self.vuer.add_handler("CONNECTION_STATUS")
        async def on_connection_status(event, session):
            status = event.value
            logger.info(f"VR Connection status changed: {status}")

            if status == "disconnected":
                logger.info("Connection lost. Attempting to reconnect...")
                try:
                    await self.vuer.reconnect()
                    logger.info("Reconnection successful!")
                except Exception as e:
                    logger.error(f"Reconnection failed: {str(e)}")
            elif status == "connected":
                logger.info("VR connection established.")

        logger.info("Event handlers registered")
        logger.info("Vuer setup complete")

        # Connection status
        self.received_hand_data = False
        self.received_camera_data = False
        self.last_status_time = time.time()
        self.status_interval = 5.0  # Print status every 5 seconds

    async def on_hand_move(self, event, session):
        try:
            if not self.received_hand_data:
                print("\n" + "="*60)
                print("First hand movement data received!")
                print("="*60)
                self.received_hand_data = True

            # Print hand data in a readable format
            print("\n" + "="*60)
            print(f"[HAND TRACKING] Event Key: {getattr(event, 'key', 'N/A')}")
            print(f"Timestamp: {time.strftime('%H:%M:%S.%f')[:-3]}")
            print("-"*60)

            # Update shared arrays and print detailed data
            if event and event.value:
                data = event.value

                # Left hand data: 400 values = 25 joints * 16 (4x4 matrix each)
                if "left" in data:
                    left_array = np.array(data["left"])
                    print("\n[Left Hand Data]")
                    print(f"  Total values: {left_array.size} (25 joints × 16 values each)")

                    # Reshape to 25 joints, each with a 4x4 matrix
                    if left_array.size == 400:
                        joints = left_array.reshape(25, 4, 4)
                        print(f"  Reshaped to: {joints.shape} (25 joints, 4x4 matrices)")

                        # Note: 25 joints total (MediaPipe has 21 landmarks, but data may include additional joints)

                        # Print wrist position (joint 0)
                        wrist_matrix = joints[0]
                        wrist_pos = wrist_matrix[:3, 3]
                        print(f"\n  [Wrist Position] (x={wrist_pos[0]:.3f}, y={wrist_pos[1]:.3f}, z={wrist_pos[2]:.3f})")

                        # Print index finger joints (joints 5-8)
                        print("\n  [Index Finger]")
                        index_joints = [5, 6, 7, 8]  # MCP, PIP, DIP, TIP
                        index_names = ["MCP", "PIP", "DIP", "TIP"]
                        for idx, name in zip(index_joints, index_names):
                            joint_matrix = joints[idx]
                            joint_pos = joint_matrix[:3, 3]
                            print(f"    {name}: (x={joint_pos[0]:.3f}, y={joint_pos[1]:.3f}, z={joint_pos[2]:.3f})")

                        # Store first joint (wrist) matrix in shared array
                        if len(self.left_hand_shared) >= 16:
                            self.left_hand_shared[:] = joints[0].flatten()[:16]

                # Left hand state
                if "leftState" in data:
                    left_state = data["leftState"]
                    print("\n[Left Hand State]")
                    print(f"  Pinch: {left_state.get('pinch', False)} (Value: {left_state.get('pinchValue', 0):.3f})")
                    print(f"  Squeeze: {left_state.get('squeeze', False)} (Value: {left_state.get('squeezeValue', 0):.3f})")
                    print(f"  Tap: {left_state.get('tap', False)} (Value: {left_state.get('tapValue', 0):.3f})")

                # Right hand data: 400 values = 25 joints * 16 (4x4 matrix each)
                if "right" in data:
                    right_array = np.array(data["right"])
                    print("\n[Right Hand Data]")
                    print(f"  Total values: {right_array.size} (25 joints × 16 values each)")

                    # Reshape to 25 joints, each with a 4x4 matrix
                    if right_array.size == 400:
                        joints = right_array.reshape(25, 4, 4)
                        print(f"  Reshaped to: {joints.shape} (25 joints, 4x4 matrices)")

                        # Print wrist position (joint 0)
                        wrist_matrix = joints[0]
                        wrist_pos = wrist_matrix[:3, 3]
                        print(f"\n  [Wrist Position] (x={wrist_pos[0]:.3f}, y={wrist_pos[1]:.3f}, z={wrist_pos[2]:.3f})")

                        # Print index finger joints (joints 5-8)
                        print("\n  [Index Finger]")
                        index_joints = [5, 6, 7, 8]  # MCP, PIP, DIP, TIP
                        index_names = ["MCP", "PIP", "DIP", "TIP"]
                        for idx, name in zip(index_joints, index_names):
                            joint_matrix = joints[idx]
                            joint_pos = joint_matrix[:3, 3]
                            print(f"    {name}: (x={joint_pos[0]:.3f}, y={joint_pos[1]:.3f}, z={joint_pos[2]:.3f})")

                        # Store first joint (wrist) matrix in shared array
                        if len(self.right_hand_shared) >= 16:
                            self.right_hand_shared[:] = joints[0].flatten()[:16]

                # Right hand state
                if "rightState" in data:
                    right_state = data["rightState"]
                    print("\n[Right Hand State]")
                    print(f"  Pinch: {right_state.get('pinch', False)} (Value: {right_state.get('pinchValue', 0):.3f})")
                    print(f"  Squeeze: {right_state.get('squeeze', False)} (Value: {right_state.get('squeezeValue', 0):.3f})")
                    print(f"  Tap: {right_state.get('tap', False)} (Value: {right_state.get('tapValue', 0):.3f})")

            print("="*60)

        except Exception as e:
            print(f"\nError processing hand movement: {e}")
            import traceback
            traceback.print_exc()
            if event:
                print("\n[Raw Event Data]")
                print(f"  Event: {event}")
                if hasattr(event, 'value'):
                    print(f"  Event.value: {event.value}")

    async def main_scene(self, session):
        print("\nMain scene started")
        print("Connection status: Active")

        # Set default scene with hand tracking enabled
        session.set @ DefaultScene(
            frameloop="always",
            handTracking=True,
            handTrackingOptions={
                "enable": True,
                "maxHands": 2,
                "modelComplexity": 1
            }
        )

        # Enable hand visualization with debug info
        logger.info("Enabling hand tracking visualization...")
        session.upsert @ Hands(
            key="hands",
            debug=True  # Enable debug visualization
        )

        print("Hand tracking enabled")
        print("Please check hand tracking status...")

        while True:
            current_time = time.time()
            if current_time - self.last_status_time >= self.status_interval:
                print("\n=== Connection Status ===")
                print(f"Hand data reception: {'Active' if self.received_hand_data else 'Inactive'}")
                print(f"Camera data reception: {'Active' if self.received_camera_data else 'Inactive'}")
                if not self.received_hand_data:
                    print("Warning: Not receiving hand tracking data.")
                    print("- Check if VR controllers are turned on")
                    print("- Check if hands are within camera view")
                self.last_status_time = current_time
            await asyncio.sleep(1/60)

    def run(self):
        try:
            print("\n=== VR Data Visualization Started ===")
            print("\n[Step 1: Local PC Tasks]")
            print("- VR server running on local PC port 8012")
            print("- Hand movement data processed on local PC")

            print("\n[Step 2: VR Headset Connection]")
            print("1. For PC browser testing:")
            print("   https://localhost:8012?grid=False")
            print("\n2. For VR headset connection:")
            print("   https://192.168.0.8:8012?ws=wss://192.168.0.8:8012")
            print("   (192.168.0.8 is your PC's local IP)")
            print("\n3. If security warning appears:")
            print("   - Click anywhere on the page and type 'thisisunsafe'")
            print("4. Click 'Virtual Reality' button")
            print("5. Allow VR permissions")
            print("6. Put on VR headset")

            print("\n[Data Flow]")
            print("VR Headset/Controllers -> Local Server (Data Processing) -> VR Display")
            print("\nNote: Connect using local IP address, not https://vuer.ai!")

            print("\nLocal server waiting...")
            print("Press Ctrl+C to exit")

            # Start Vuer server
            self.vuer.run()

        except KeyboardInterrupt:
            print("\nShutting down...")

if __name__ == "__main__":
    visualizer = VRVisualizer()
    visualizer.run()
