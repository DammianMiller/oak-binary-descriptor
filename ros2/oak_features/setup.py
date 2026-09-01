import os
from glob import glob

from setuptools import setup

package_name = "oak_features"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Luxonis",
    maintainer_email="support@luxonis.com",
    description="ROS2 node publishing keypoints with binary descriptors from an OAK camera.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "feature_node = oak_features.feature_node:main",
        ],
    },
)
