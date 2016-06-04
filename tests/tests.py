# coding=utf-8
from unittest import TestCase

import sys

import findschemes, evaluate


class BaseTestCase(TestCase):

    def setUp(self):
        self.endings_file = 'sample.pgold'
        self.init_type = 'o'
        self.output_file = 'out.txt'
        args = [
            self.endings_file,
            self.init_type,
            self.output_file,
        ]
        findschemes.main(args)


class FindschemesTestCase(BaseTestCase):


    def test_findschemes(self):
        with open(self.output_file, 'r') as f:
            output = f.read()
            self.assertEqual(output, """\
schwoll daran ruhevoll hinan lauscht empor rauscht hervor
1 2 1 2 3 4 3 4

ihm brut menschenlist todesglut ist grund bist gesund
1 2 1 2 3 4 3 4

nicht meer gesicht her nicht blau angesicht tau
1 2 1 2 3 4 3 4

schwoll fuß sehnsuchtsvoll gruß ihm geschehn hin gesehn
1 2 1 2 3 4 3 4

""")


class EvaluateTestCase(BaseTestCase):

    def setUp(self):
        super(EvaluateTestCase, self).setUp()
        args = [
            self.endings_file,
            self.output_file,
        ]
        evaluate.main(args)

    def test_evaluate(self):
        if not hasattr(sys.stdout, "getvalue"):
            self.fail("Test needs to run in buffered mode")
        output = sys.stdout.getvalue().strip()
        self.assertTrue("""\
Num of stanzas:  4
Num of lines:  32
Num of end word types:  29

Naive baseline:
Accuracy 0.0 4.0 0.0
Precision 0.25
Recall 0.5
F-score 0.333333333333

Less naive baseline:
Accuracy 4.0 4.0 100.0
Precision 1.0
Recall 1.0
F-score 1.0

out.txt :
Accuracy 4.0 4.0 100.0
Precision 1.0
Recall 1.0
F-score 1.0"""
    in output)
