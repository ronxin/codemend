from setuptools import setup

def readme():
  with open('README.rst') as f:
    return f.read()

setup(name='codemend',
      version='0.1',
      description='CodeMend Demo',
      long_description=readme(),
      keywords='nlp code',
      url='http://github.com/ronxin/codemend',
      author='Xin Rong',
      author_email='ronxin@umich.edu',
      license='MIT',
      packages=['codemend'],
      install_requires=[
          'astunparse >= 1.3.0',
          'gensim >= 0.12.4',
          'numpy >= 1.3',
          'Pattern >= 2.6',
          'scipy >= 0.7.0',
          'matplotlib >= 1.4.0',
          'Whoosh >= 2.7.0',
          'redbaron >= 0.5.1',
          'numpydoc >= 0.6.0',
          'jedi >= 0.9.0',
          'requests >= 2.8',
          'recordclass >= 0.4',
          #'googlesearch >= 0.7',
          'funcsigs >= 0.4',
          'seaborn >= 0.7',
      ],
      scripts=['bin/codemend', 'bin/checkdoc', 'bin/cb'],
      include_package_data=True)
