from setuptools import setup, find_packages
setup(
        name='Brain'
        version = '0.1.0'
        description = 'Emulates the brain'
        packages = find_packages()
        install_requires = ['wrapt']
        test_require = ['pytest']
)
