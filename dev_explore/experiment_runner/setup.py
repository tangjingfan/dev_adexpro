from glob import glob
from setuptools import find_packages, setup

package_name = 'experiment_runner'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools', 'pyyaml'],
    zip_safe=True,
    maintainer='richard',
    maintainer_email='ruizhi_deng@sjtu.edu.cn',
    description='Automated batch runner for exploration baselines.',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'runner = experiment_runner.runner:main',
        ],
    },
)
