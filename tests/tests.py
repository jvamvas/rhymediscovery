# coding=utf-8
from __future__ import division

from unittest import TestCase, skipUnless

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

    @skipUnless('test_all_rhymedata' in ' '.join(sys.argv), 'skip all.pgold')
    def test_all_rhymedata(self):
        args = [
            '../rhymedata/english_gold/all.pgold',
            'o',
            'out_all.txt',
        ]
        findschemes.main(args)

    @skipUnless('test_sidney_rhymedata' in ' '.join(sys.argv), 'skip sidney.pgold')
    def test_sidney_rhymedata(self):
        input_file = '../rhymedata/english_gold/sidney.pgold'
        output_file = 'out_sidney.txt'
        args = [
            input_file,
            'o',
            output_file,
        ]
        findschemes.main(args)
        evaluate_args = [
            input_file,
            output_file,
        ]
        evaluate.main(evaluate_args)

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

    def test_main(self):
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
        with open(self.endings_file, 'r') as f:
            gstanzaschemes, gstanzas = evaluate.load_gold(f)
        self.results = findschemes.find_schemes(self.stanzas, findschemes.init_basicortho_ttable)
        hstanzaschemes = [scheme for (words, scheme) in self.results]
        result = evaluate.evaluate(gstanzaschemes, gstanzas, hstanzaschemes)
        self.assertEqual(result.num_stanzas, 4)
        self.assertEqual(result.num_lines, 32)
        self.assertEqual(result.num_end_word_types, 29)
        self.assertEqual(result.naive_baseline_success.accuracy, [0.0, 4.0, 0.0])
        self.assertEqual(result.naive_baseline_success.precision, 0.25)
        self.assertEqual(result.naive_baseline_success.recall, 0.5)
        self.assertAlmostEqual(result.naive_baseline_success.f_score, 1/3)
