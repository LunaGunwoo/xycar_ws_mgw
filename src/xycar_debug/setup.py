import os
from glob import glob

from setuptools import find_packages, setup


package_name = "xycar_debug"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        ("share/" + package_name, ["package.xml"]),
        (
            os.path.join("share", package_name, "launch"),
            glob("launch/*.launch.py"),
        ),
        (
            os.path.join("share", package_name, "config"),
            glob("config/*.yaml"),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Gunwoo Moon",
    maintainer_email="moongunwoo7019@naver.com",
    description="LiDAR cone Viewer with Space-key debug driving for Xycar.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "cone_path_viewer = xycar_debug.cone_path_viewer:main",
        ],
    },
)
