import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'cogniplan'


def optional_files(pattern):
    return glob(pattern) if glob(pattern) else []


setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
        (os.path.join('share', package_name, 'mapinpaint'), glob('cogniplan/mapinpaint/*.yaml')),
        (os.path.join('share', package_name, 'checkpoints', 'cogniplan_exp_pred7'),
         optional_files('checkpoints/cogniplan_exp_pred7/*')),
        (os.path.join('share', package_name, 'checkpoints', 'wgan_inpainting'),
         optional_files('checkpoints/wgan_inpainting/*')),
        (os.path.join('share', package_name, 'model'),
         optional_files('cogniplan/model/*.pth') + optional_files('cogniplan/model/*.pt') +
         optional_files('cogniplan/model/*.yaml')),
        (os.path.join('lib', package_name), ['scripts/cogniplan_planner_env']),
    ],
    install_requires=['setuptools', 'sensor_msgs_py'],
    zip_safe=True,
    maintainer='robot',
    description='CogniPlan ROS2 planner baseline',
    license='MIT License',
    entry_points={
        'console_scripts': [
            'cogniplan_planner = cogniplan.cogniplan_planner:main',
        ],
    },
)
