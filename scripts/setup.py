from setuptools import setup, find_packages, Distribution

class BinaryDistribution(Distribution):
    def has_ext_modules(skip):
        return True

setup(
    name='Caffe2',
    version='0.0.1',
    description='An machine learning engine.',
    url='http://caffe2.ai',
    author='Yangqing Jia',
    license='BSD',
    classifiers=[
        'Development Status :: 3 - Alpha',

        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',

        'License :: OSI Approved :: BSD License',

        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        #'Programming Language :: Python :: 3',
        #'Programming Language :: Python :: 3.3',
        #'Programming Language :: Python :: 3.4',
        #'Programming Language :: Python :: 3.5',
    ],
    keywords='neural network machine learning caffe2 caffe',
    install_requires=[''],
    extras_require={},
    packages=find_packages(),
    package_data={
        'caffe2': ['python/caffe2_pybind11_state.so', 'libCaffe2_CPU.so'],
        },
    distclass=BinaryDistribution
)
