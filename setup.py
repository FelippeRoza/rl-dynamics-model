from setuptools import setup, find_packages

setup(name='costDynamicsModel',
      version='0.1',
      description='cost dynamics model based on mbrl-lib for safe RL applications',
      url='https://github.com/FelippeRoza/rl-dynamics-model',
      author='Felippe Roza',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
      install_requires=['gymnasium>=0.26.3', 'mbrl>=0.2.0',]
)
