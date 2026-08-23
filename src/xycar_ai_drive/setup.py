import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'xycar_ai_drive'


setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name],
        ),
        ('share/' + package_name, ['package.xml']),
        (
            os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py'),
        ),
        (
            os.path.join('share', package_name, 'config'),
            glob('config/*.yaml'),
        ),
    ],
    install_requires=['setuptools'],
    tests_require=['pytest'],
    zip_safe=True,
    maintainer='Gunwoo Moon',
    maintainer_email='moongunwoo7019@naver.com',
    description='TorchScript front-camera policy with safe Xycar control.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            (
                'front_cam_policy = '
                'xycar_ai_drive.front_cam_policy_node:main'
            ),
            (
                'front_cam_policy_gpu_server = '
                'xycar_ai_drive.policy_ipc:main'
            ),
            (
                'traffic_shortcut_gpu_server = '
                'xycar_ai_drive.dual_policy_ipc:main'
            ),
            (
                'traffic_shortcut_policy = '
                'xycar_ai_drive.traffic_shortcut_policy_node:main'
            ),
            (
                'guided_policy_collector = '
                'xycar_ai_drive.guided_policy_collector:main'
            ),
            (
                'competition_policy = '
                'xycar_ai_drive.competition_policy_node:main'
            ),
            (
                'competition_policy_gpu_server = '
                'xycar_ai_drive.competition_ipc:main'
            ),
            (
                'competition_replay = '
                'xycar_ai_drive.competition_replay:main'
            ),
            (
                'live_warp_tuner = '
                'xycar_ai_drive.live_warp_tuner:main'
            ),
            (
                'traffic_light_viewer = '
                'xycar_ai_drive.traffic_light_viewer:main'
            ),
            (
                'traffic_shortcut_monitor = '
                'xycar_ai_drive.traffic_shortcut_monitor:main'
            ),
        ],
    },
)
