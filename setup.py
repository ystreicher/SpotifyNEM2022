from setuptools import setup, find_packages

console_scripts = [
    #'climateGAN-train=climateGAN.cli.entry_points:train_main',
]

setup(
    name='nem',
    description='Nothing Else Matters',
    version='1.0',
    packages=['nem'],
    entry_points = {
        'console_scripts': console_scripts
    }
)