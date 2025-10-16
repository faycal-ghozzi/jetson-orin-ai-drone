from launch import LaunchDescription
from launch_ros.actions import Node
import os

def generate_launch_description():

    yolo_env = os.environ.copy()
    yolo_env["LD_LIBRARY_PATH"] = "/home/reisar/.ai-drone/lib/python3.10/site-packages/nvidia/cusparselt/lib:" + yolo_env.get("LD_LIBRARY_PATH", "")
    
    return LaunchDescription([
        Node(package='ai_drone', executable='video_file', name='video_file',
            parameters=[{
                'file': '/home/reisar/ai-drone-ws/media/demo.mp4',
                'loop': True,
                'resize_w': 360,
                'resize_h': 640,
                'rotate_deg': 0,
                'mirror': False,
                'speed': 1.0,
                'jpeg_quality': 100,
                'pub_raw': '/camera/image',
                'pub_comp': '/camera/image/compressed',
            }]
        ),
        Node(package='ai_drone', executable='yolo_trt_node',     name='yolo', respawn=True, respawn_delay=2.0),
        Node(package='ai_drone', executable='tracker_node',       name='tracker', respawn=True, respawn_delay=2.0),
        Node(package='ai_drone', executable='process_perception', name='perception', respawn=True, respawn_delay=2.0),
        Node(package='ai_drone', executable='overlay_node',       name='overlay', respawn=True, respawn_delay=2.0),
        Node(package='ai_drone', executable='telemetry_acquire',  name='telemetry', respawn=True, respawn_delay=2.0),
        Node(package='ai_drone', executable='flask_streamer',     name='flask', respawn=True, respawn_delay=2.0,
             parameters=[{'topic':'/camera/overlay/compressed','quality':80}]),
    ])

