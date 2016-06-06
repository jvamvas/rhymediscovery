# coding=utf-8
from unittest import TestCase

import sys

import numpy

import findschemes
import evaluate


class BaseTestCase(TestCase):
    def setUp(self):
        self.endings_file = 'sample.pgold'
        self.init_type = 'o'
        self.output_file = 'out.txt'
        with open(self.endings_file, 'r') as f:
            self.stanzas = findschemes.load_stanzas(f)


class FindschemesTestCase(BaseTestCase):
    def setUp(self):
        super(FindschemesTestCase, self).setUp()

    def test_main(self):
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

    def test_basicortho_findschemes(self):
        results = findschemes.find_schemes(self.stanzas, findschemes.init_basicortho_ttable)
        self.assertEqual(results[0], (
            ['schwoll', 'daran', 'ruhevoll', 'hinan', 'lauscht', 'empor', 'rauscht', 'hervor'],
            [1, 2, 1, 2, 3, 4, 3, 4],
        ))
        for stanza, scheme in results:
            self.assertEqual(scheme, [1, 2, 1, 2, 3, 4, 3, 4])

    def test_uniform_init_findschemes(self):
        results = findschemes.find_schemes(self.stanzas, findschemes.init_uniform_ttable)
        self.assertEqual(results[0], (
            ['schwoll', 'daran', 'ruhevoll', 'hinan', 'lauscht', 'empor', 'rauscht', 'hervor'],
            [1, 2, 1, 2, 3, 4, 3, 4],
        ))
        self.assertNotEqual(results[2][1], [1, 2, 1, 2, 3, 4, 3, 4], msg='Uniform misclassifies third stanza')

    def test_get_wordset(self):
        stanzas = [['word1a', 'word1b'], ['word2a', 'word2b']]
        words = ['word1a', 'word1b', 'word2a', 'word2b']
        self.assertEqual(findschemes.get_wordset(stanzas), words)

    def test_init_uniform_table(self):
        words = ['word1', 'word2']
        t_table = numpy.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
        numpy.testing.assert_array_equal(findschemes.init_uniform_ttable(words), t_table)

    def test_get_rhymelists(self):
        stanza = ['w1', 'w2', 'w3', 'w4']
        scheme1 = [1, 2, 1, 2]
        self.assertEqual(findschemes.get_rhymelists(stanza, scheme1), [['w1', 'w3'], ['w2', 'w4']])
        scheme2 = [1, 1, 2, 2]
        self.assertEqual(findschemes.get_rhymelists(stanza, scheme2), [['w1', 'w2'], ['w3', 'w4']])


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
