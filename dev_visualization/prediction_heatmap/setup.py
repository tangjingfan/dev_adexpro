from glob import glob
from setuptools import find_packages, setup

package_name = 'prediction_heatmap'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jingfan',
    maintainer_email='tangjingfan@gmail.com',
    description='Live heatmaps of map-prediction accuracy.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'prediction_heatmap_node = prediction_heatmap.heatmap_node:main',
        ],
    },
)
