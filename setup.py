from setuptools import setup
setup(
    name='pysigprofiler',
    version='0.1',
    description='Python implementation of sigprofiler',
    author='Jason Fan',
    license='MIT',
    packages=['pysigprofiler'],
    install_requires=[
        'scikit-learn',
        'numpy'
    ],
    zip_safe=False,
    test_suite='nose.collector',
    tests_require=['nose']
)