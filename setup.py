from setuptools import setup

setup(name='word2veclite',
      version='0.1.0',
      description='Python implementation of word2vec',
      url='https://github.com/cbellei/word2veclite',
      author='Claudio Bellei',
      author_email='',
      license='OSI Approved Apache Software License',
      packages=['word2veclite'],
      zip_safe=False,
      test_suite='nose.collector',
      tests_require=['nose']
      )
