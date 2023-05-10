from setuptools import setup, find_packages

setup(
    name='qualitylib',
    author='Abhinau Kumar',
    author_email='ab.kumr98@gmail.com',
    version='0.1.0',
    url='https://github.com/abhinaukumar/qualitylib',
    description='Package for seamlessly running quality assessment experiments in Python.',
    install_requires=[
        'joblib',
        'numpy',
        'scipy',
        'scikit-learn',
        'videolib @ git+https://github.com/abhinaukumar/videolib@main'
    ],
    python_requires='>=3.7.0',
    license='MIT License',
    packages=find_packages()
)
