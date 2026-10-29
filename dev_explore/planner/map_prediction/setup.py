from setuptools import find_packages, setup

package_name = 'map_prediction'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/models', ['models/best_fpunet_256_8.pth']),
        ('share/' + package_name + '/config', [
            'config/map_prediction.yaml',
            'config/online_update.yaml',
            'config/oracle_map_prediction.yaml',
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Jingfan',
    maintainer_email='tangjingfan@gmail.com',
    description='Map prediction node with a learned (FPUnet) and an oracle backend.',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'map_prediction_node = map_prediction.map_prediction_node:main',
            'self_adapt_node = map_prediction.self_adapt:main',
            'oracle_map_prediction_node = map_prediction.oracle_map_prediction_node:main',
        ],
    },
)
