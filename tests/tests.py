# coding=utf-8
from unittest import TestCase

import sys

import findschemes
import evaluate


class BaseTestCase(TestCase):
    def setUp(self):
        self.endings_file = 'sample.pgold'
        self.init_type = 'o'
        self.output_file = 'out.txt'


class FindschemesTestCase(BaseTestCase):
    def setUp(self):
        super(FindschemesTestCase, self).setUp()

    def test_findschemes(self):
        args = [
            self.endings_file,
            self.init_type,
            self.output_file,
        ]
        findschemes.main(args)
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

    def test_uniform_init_findschemes(self):
        args = [
            self.endings_file,
            'u',
            self.output_file,
        ]
        findschemes.main(args)
        with open(self.output_file, 'r') as f:
            output = f.read()
            self.assertEqual(output, """\
schwoll daran ruhevoll hinan lauscht empor rauscht hervor
1 2 1 2 3 4 3 4

ihm brut menschenlist todesglut ist grund bist gesund
1 2 1 2 3 4 3 4

nicht meer gesicht her nicht blau angesicht tau
1 2 2 1 3 4 3 4

schwoll fuß sehnsuchtsvoll gruß ihm geschehn hin gesehn
1 2 1 2 3 4 3 4

""")


class EvaluateTestCase(BaseTestCase):
    def setUp(self):
        super(EvaluateTestCase, self).setUp()
        findscheme_args = [
            self.endings_file,
            self.init_type,
            self.output_file,
        ]
        findschemes.main(findscheme_args)
        evaluate_args = [
            self.endings_file,
            self.output_file,
        ]
        evaluate.main(evaluate_args)

    def test_evaluate(self):
        if not hasattr(sys.stdout, "getvalue"):
            self.fail("Test needs to run in buffered mode")
        output = sys.stdout.getvalue().strip()
        self.assertIn("""\
Num of stanzas:  4
Num of lines:  32
Num of end word types:  29""", output)
