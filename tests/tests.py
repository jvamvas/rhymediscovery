# coding=utf-8
from __future__ import division

from unittest import TestCase

import numpy

from rhymediscovery import find_schemes, evaluate_schemes


class BaseTestCase(TestCase):
    def setUp(self):
        self.endings_file = 'tests/data/sample.pgold'
        self.output_file = 'tests/out.txt'
        with open(self.endings_file, 'r') as f:
            self.stanzas = find_schemes.load_stanzas(f)


class FindSchemesTestCase(BaseTestCase):
    def setUp(self):
        super(FindSchemesTestCase, self).setUp()

    def test_main(self):
        args = [
            self.endings_file,
            self.output_file,
            '-t', 'd',
        ]
        find_schemes.main(args)
        with open(self.output_file, 'r') as f:
            output = f.read()
            self.assertTrue(output.startswith("""\
minds love finds remove
1 2 1 2
"""))

    def test_basicortho_find_schemes(self):
        results = find_schemes.find_schemes(self.stanzas, find_schemes.init_basicortho_ttable)
        self.assertEqual(results[0], (
            ('minds', 'love', 'finds', 'remove'),
            (1, 2, 1, 2),
        ))

    def test_uniform_init_find_schemes(self):
        results = find_schemes.find_schemes(self.stanzas, find_schemes.init_uniform_ttable)
        self.assertEqual(results[0], (
            ('minds', 'love', 'finds', 'remove'),
            (1, 2, 2, 1),
        ))

    def test_get_wordset(self):
        stanzas = [find_schemes.Stanza(['word1a', 'word1b']), find_schemes.Stanza(['word2a', 'word2b'])]
        words = ['word1a', 'word1b', 'word2a', 'word2b']
        self.assertEqual(find_schemes.get_wordlist(stanzas), words)

    def test_init_uniform_table(self):
        words = ['word1', 'word2']
        t_table = numpy.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
        numpy.testing.assert_array_equal(find_schemes.init_uniform_ttable(words), t_table)

    def test_get_rhymelists(self):
        words = ['w1', 'w2', 'w3', 'w4']
        stanza = find_schemes.Stanza(words)
        stanza.set_word_indices(words)
        scheme1 = (1, 2, 1, 2)
        self.assertEqual(find_schemes.get_rhymelists(stanza, scheme1), [[0, 2], [1, 3]])
        scheme2 = (1, 1, 2, 2)
        self.assertEqual(find_schemes.get_rhymelists(stanza, scheme2), [[0, 1], [2, 3]])


class EvaluateTestCase(BaseTestCase):
    def setUp(self):
        super(EvaluateTestCase, self).setUp()

    def test_main(self):
        findscheme_args = [
            self.endings_file,
            self.output_file,
        ]
        find_schemes.main(findscheme_args)
        evaluate_args = [
            self.output_file,
            self.endings_file,
        ]
        evaluate_schemes.main(evaluate_args)

    def test_evaluate(self):
        with open(self.endings_file, 'r') as f:
            gstanzaschemes, gstanzas = evaluate_schemes.load_gold(f)
        self.results = find_schemes.find_schemes(self.stanzas, find_schemes.init_difflib_ttable)
        result = evaluate_schemes.evaluate(gstanzaschemes, gstanzas, self.results)
        self.assertEqual(result.num_stanzas, 4)
        self.assertEqual(result.num_lines, 14)
        self.assertEqual(result.num_end_word_types, 14)
        self.assertAlmostEqual(result.naive_baseline_success.accuracy, 1)
        self.assertAlmostEqual(result.naive_baseline_success.precision, 1)
        self.assertAlmostEqual(result.naive_baseline_success.recall, 1)
        self.assertAlmostEqual(result.naive_baseline_success.f_score, 1)
        self.assertAlmostEqual(result.less_naive_baseline_success.accuracy, 1)
        self.assertAlmostEqual(result.less_naive_baseline_success.precision, 1)
        self.assertAlmostEqual(result.less_naive_baseline_success.recall, 1)
        self.assertAlmostEqual(result.less_naive_baseline_success.f_score, 1)
        self.assertGreaterEqual(result.input_success.accuracy, 0.75)
        self.assertGreaterEqual(result.input_success.precision, 0.7)
        self.assertGreaterEqual(result.input_success.recall, 0.7)
        self.assertGreaterEqual(result.input_success.f_score, 0.7)


class ParseSchemesTestCase(TestCase):

    def setUp(self):
        self.scheme_filename = 'rhymediscovery/data/schemes.json'
        with open(self.scheme_filename, 'r') as f:
            self.schemes = find_schemes.Schemes(f)

    def test_scheme_list(self):
        self.assertEqual(len(self.schemes.scheme_list), 462)
        self.assertEqual(self.schemes.scheme_list[0], (1, 1))
        self.assertEqual(self.schemes.num_schemes, 462)

    def test_get_schemes_for_len(self):
        self.assertEqual(self.schemes.get_schemes_for_len(2), [0])


class StanzaTestCase(TestCase):

    def setUp(self):
        self.words = ['w1', 'w2', 'w3']
        self.stanza = find_schemes.Stanza(self.words)

    def test_str(self):
        self.assertEqual(str(self.stanza), 'w1 w2 w3')
