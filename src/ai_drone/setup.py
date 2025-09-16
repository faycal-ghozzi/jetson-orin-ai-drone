from setuptools import setup, find_packages
package_name = 'ai_drone'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/ai_drone']),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/bringup.launch.py']),
        ('share/' + package_name, ['ai_drone/.env']),
        ('share/' + package_name + '/models', []),
    ],
    install_requires=[],
    zip_safe=True,
    maintainer='ai-drone',
    maintainer_email='you@example.com',
    description='RTSP + TensorRT YOLOv8 + Flask streamer',
    license='MIT',
    entry_points={
        'console_scripts': [
            'video_acquire_ros2 = ai_drone.video_acquire_ros2:main',
            'yolo_trt_node     = ai_drone.yolo_trt_node:main',
            'tracker_node      = ai_drone.tracker_node:main',
            'flask_streamer    = ai_drone.flask_streamer:main',
            'overlay_node      = ai_drone.overlay_node:main',
            'telemetry_acquire = ai_drone.telemetry_acquire:main',
            'process_perception = ai_drone.process_perception:main',
        ],

    },
)

