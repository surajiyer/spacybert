from pathlib import Path
from setuptools import setup, find_packages

package_name = 'spacybert'
root = Path(__file__).parent.resolve()

# Read in package meta from about.py
about_path = root / package_name / 'about.py'
with about_path.open('r', encoding='utf8') as f:
    about = {}
    exec(f.read(), about)

# Get readme
readme_path = root / 'README.md'
with readme_path.open('r', encoding='utf8') as f:
    readme = f.read()

install_requires = [
    'torch==1.4.0+cpu',
    'transformers>=3.0.0',
    'spacy>=2.2.1,<3.0.0',
    'spacy-langdetect>=0.1.2'
]
test_requires = ['pytest']

setup(
    name=package_name,
    description=about['__summary__'],
    long_description=readme,
    long_description_content_type='text/markdown',
    author=about['__author__'],
    author_email=about['__email__'],
    url=about['__uri__'],
    version=about['__version__'],
    license=about['__license__'],
    packages=find_packages(),
    install_requires=install_requires,
    test_requires=test_requires,
    zip_safe=False,
)
