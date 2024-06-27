from setuptools import setup


def readme():
    with open('README.md') as f:
        return f.read()


setup(name='multiset_canonical_correlation_analysis',
      version='0.0.1',
      description='Multiset Canonical Correlation Analysis (mCCA) implementation',
      long_description=readme(),
      long_description_content_type='text/markdown',
      classifiers=[
          'Programming Language :: Python :: 3',
          'License :: OSI Approved :: MIT License',
          'Operating System :: OS Independent',
      ],
      keywords='multiset canonical correlation analysis, multiset_canonical_correlation_analysis, bss, multiset analysis',
      url='https://github.com/SSTGroup/independent_vector_analysis',
      author='Isabell Lehmann',
      author_email='isabell.lehmann@sst.upb.de',
      license='LICENSE',
      packages=['multiset_canonical_correlation_analysis'],
      python_requires='>=3.6',
      install_requires=[
          'numpy',
          'pandas',
          'scipy',
          'pytest',
          'matplotlib',
          'independent_vector_analysis',
          'Jinja2'  # for df.to_latex()
      ],
      include_package_data=True,  # to include non .py-files listed in MANIFEST.in
      zip_safe=False)
