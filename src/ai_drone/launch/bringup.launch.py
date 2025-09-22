from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(package='ai_drone', executable='video_file', name='video_file',
            parameters=[{
                'file': '~/ai-drone-ws/media/demo.mp4',  # <— change path if needed
                'loop': True,
                'resize_w': 640,
                'resize_h': 480,
                'rotate_deg': 0,       # set 270 if your test video is sideways
                'mirror': False,
                'speed': 1.0,          # 1x real-time
                'jpeg_quality': 80,
                'pub_raw': '/camera/image',
                'pub_comp': '/camera/image/compressed',
            }]
        ),
        Node(package='ai_drone', executable='video_acquire_ros2', name='video', respawn=True, respawn_delay=2.0),
        Node(package='ai_drone', executable='yolo_trt_node',     name='yolo', respawn=True, respawn_delay=2.0),
        Node(package='ai_drone', executable='tracker_node',       name='tracker', respawn=True, respawn_delay=2.0),
        Node(package='ai_drone', executable='process_perception', name='perception', respawn=True, respawn_delay=2.0),
        Node(package='ai_drone', executable='overlay_node',       name='overlay', respawn=True, respawn_delay=2.0),
        Node(package='ai_drone', executable='telemetry_acquire',  name='telemetry', respawn=True, respawn_delay=2.0),
        Node(package='ai_drone', executable='flask_streamer',     name='flask', respawn=True, respawn_delay=2.0,
             parameters=[{'topic':'/camera/overlay/compressed','quality':80}]),
    ])

