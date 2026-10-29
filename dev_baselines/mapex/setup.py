from setuptools import find_packages, setup
from glob import glob
import os

package_name = "mapex"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "rviz"), glob("rviz/*.rviz")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="richard",
    maintainer_email="ruizhi_deng@sjtu.edu.cn",
    description="TODO: Package description",
    license="TODO: License declaration",
    extras_require={},
    entry_points={
        "console_scripts": [
            "mapex_node = mapex.mapex_node:main",
        ],
    },
)
