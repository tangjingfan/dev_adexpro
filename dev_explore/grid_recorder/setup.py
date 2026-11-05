from setuptools import find_packages, setup

package_name = 'grid_recorder'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jingfan',
    maintainer_email='tangjingfan@gmail.com',
    description='Records the size of the occupancy grid over a run.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'grid_recorder = grid_recorder.grid_recorder:main',
            'make_reference = grid_recorder.make_reference:main',
            'reference_publisher = grid_recorder.reference_publisher:main',
        ],
    },
)
