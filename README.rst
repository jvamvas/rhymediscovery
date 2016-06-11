.. image:: https://travis-ci.org/jvamvas/rhymediscovery.svg?branch=master
    :target: https://travis-ci.org/jvamvas/rhymediscovery

rhymediscovery
--------------

A python package for detecting rhyme schemes in poetry. With standard configuration, it achieves about 65% accuracy in the `rhymedata <https://github.com/sravanareddy/rhymedata>`_ corpus.

Basic usage
===========

.. code-block::

    pip install rhymediscovery

.. code-block:: python

   >>> from rhymediscovery import find_schemes
   >>> find_schemes.find_schemes([['herz', 'welt', 'geld', 'schmerz'], ...])
   [(('herz', 'welt', 'geld', 'schmerz'), (1, 2, 1, 2)), ...]


Command line
============

The command line tool :code:`find_schemes` expects a file formatted like the `.pgold` files in the `rhymedata <https://github.com/sravanareddy/rhymedata>`_ corpus. A sample file can be found `here in the repo <https://github.com/jvamvas/rhymediscovery/blob/master/tests/data/sample.pgold>`_.

To demonstrate the tool, we clone the `rhymedata <https://github.com/sravanareddy/rhymedata>`_ corpus:

.. code-block::

    git clone https://github.com/sravanareddy/rhymedata.git

Then we run:

.. code-block::

    find_schemes rhymedata/english_gold/1415.pgold 1415.out

The :code:`1415.pgold` file from the corpus has been manually annotated with the correct rhyme schemes. Thus we can evaluate our result file  using:

.. code-block:: bash

    evaluate_schemes 1415.out rhymedata/english_gold/1415.pgold


Credits
=======
This package implements the following paper:

* Sravana Reddy and Kevin Knight. "Unsupervised discovery of rhyme schemes." Proceedings of ACL 2011. (`pdf <http://cs.wellesley.edu/~sravana/papers/rhymes_acl.pdf>`_)

It is a fork of the reference implementation by `@sravanareddy <https://github.com/sravanareddy>`_ which is available in its original form `here <https://github.com/sravanareddy/rhymediscovery>`_.
