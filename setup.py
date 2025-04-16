from setuptools import setup, find_packages

setup(
    name='LoRe-Genome',
    version='0.1.0',
    author='Thomas Hansen',
    author_email='thomas.hansen@queensu.ca',
    description='A pipeline for protein classification using NCBI datasets.',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'logging',
        'numpy',
        'pandas',
        'pydantic',
        'pyyaml',
        'requests',
        'ncbi-datasets-pylib',
        'click',
    ],
    entry_points={
        "console_scripts": [
            # 'lore' command will call the cli() group in cli/main.py
            "lore = cli.main:cli",
        ],
    },
    classifiers=[
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)