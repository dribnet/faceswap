from setuptools import setup
from setuptools import find_packages

install_requires = [
    'numpy',
]

setup(name='faceswap',
      version='0.1.0',
      description='Routines for faceswapping images',
      author='Tom White',
      author_email='tom@sixdozen.com',
      url='https://github.com/dribnet/faceswap',
      download_url='https://github.com/dribnet/faceswap/tarball/0.1.0',
      license='MIT',
      package_data={'': ['faceswap/standard_landmarks.npy', 'faceswap/shape_predictor_68_face_landmarks.dat']},
      entry_points={
          # 'console_scripts': ['neupup = neupup.neupup:main']
      },
      install_requires=install_requires,
      packages=find_packages())
