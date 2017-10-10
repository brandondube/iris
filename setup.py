from setuptools import setup

setup(
    name='pyphase',
    version='0.0.1',
    description='A python-based phase retrieval module',
    long_description='',
    license='MIT',
    author='Brandon Dube',
    author_email='brandondube@gmail.com',
    packages=['pyphase'],
    install_requires=['numpy', 'matplotlib', 'scipy'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ]
)
