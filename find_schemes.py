#!/usr/bin/env python

"""EM algorithm for learning rhyming words and rhyme schemes with independent stanzas.
Sravana Reddy (sravana@cs.uchicago.edu), 2011.
"""

from __future__ import division, print_function, unicode_literals

import argparse
import json
import logging
import os
import sys
from collections import defaultdict, OrderedDict
from difflib import SequenceMatcher

import numpy

import celex


def load_stanzas(stanzas_file):
    """Load raw stanzas from gold standard file"""
    f = stanzas_file.readlines()
    stanzas = []
    for i, line in enumerate(f):
        if i % 4 == 0:
            stanza_words = line.strip().split()[1:]
            stanzas.append(Stanza(stanza_words))
    return stanzas


class Stanza:

    def __init__(self, stanza_words):
        self.words = tuple(stanza_words)
        self.word_indices = None

    def set_word_indices(self, words):
        self.word_indices = [words.index(word) for word in self.words]

    def __str__(self):
        return ' '.join(self.words)

    def __len__(self):
        return len(self.words)


class Schemes:

    def __init__(self, scheme_file):
        self.scheme_file = scheme_file
        # Use redundant data structures for lookup optimization
        self.scheme_list, self.scheme_dict = self._parse_scheme_file()
        self.num_schemes = len(self.scheme_list)
        self.scheme_array = self._create_scheme_array()

    def _parse_scheme_file(self):
        schemes = json.loads(self.scheme_file.read(), object_pairs_hook=OrderedDict)
        scheme_list = []
        scheme_dict = defaultdict(list)
        for scheme_len, scheme_group in schemes.items():
            for scheme_str, _count in scheme_group:
                scheme_code = tuple(int(c) for c in scheme_str.split(' '))
                scheme_list.append(scheme_code)
                scheme_dict[int(scheme_len)].append(len(scheme_list) - 1)
        return scheme_list, scheme_dict

    def _create_scheme_array(self):
        return numpy.arange(self.num_schemes)

    def get_schemes_for_len(self, n):
        """
        :return: List of indices of schemes with length n
        """
        return self.scheme_dict[n]


def get_wordlist(stanzas):
    """
    Get an iterable of all final words in all stanzas
    """
    wordlist = sorted(list(set().union(*[stanza.words for stanza in stanzas])))
    return wordlist


def get_rhymelists(stanza, scheme):
    """
    Transform stanza into list of ordered lists of rhymesets as defined by rhyme scheme
    """
    rhymelists = defaultdict(list)
    for rhyme_group, word_index in zip(scheme, stanza.word_indices):
        rhymelists[rhyme_group].append(word_index)
    return list(rhymelists.values())


def init_distance_ttable(words, distance_function):
    """
    Initialize probabilities according to a measure of orthographic similarity
    """
    n = len(words)
    t_table = numpy.zeros((n, n + 1))

    # Initialize P(c|r) accordingly
    for r, w in enumerate(words):
        for c, v in enumerate(words):
            if c < r:
                t_table[r, c] = t_table[c, r]  # Similarity is symmetric
            else:
                t_table[r, c] = distance_function(w, v) + 0.001  # For backoff
    t_table[:, n] = numpy.random.rand(1, n)  # No estimate for P(r|no history)

    # Normalize
    t_totals = numpy.sum(t_table, axis=0)
    for i, t_total in enumerate(t_totals.tolist()):
        t_table[:, i] /= t_total
    return t_table


def init_uniform_ttable(words):
    """initialize (normalized) theta uniformly"""
    n = len(words)
    return numpy.ones((n, n + 1)) * (1 / n)


def basic_word_sim(word1, word2):
    """
    Simple measure of similarity: Number of letters in common / max length
    """
    return sum([1 for c in word1 if c in word2]) / max(len(word1), len(word2))


def init_basicortho_ttable(words):
    return init_distance_ttable(words, basic_word_sim)


def difflib_similarity(word1, word2):
    sequence_matcher = SequenceMatcher(None, word1, word2)
    return sequence_matcher.ratio()


def init_difflib_ttable(words):
    return init_distance_ttable(words, difflib_similarity)


def post_prob_scheme(t_table, words, stanza, scheme):
    """
    Compute posterior probability of a scheme for a stanza, with probability of every word in rhymelist
    rhyming with all the ones before it
    """
    myprob = 1
    n = len(words)
    rhymelists = get_rhymelists(stanza, scheme)
    for rhymelist in rhymelists:
        for i, word_index in enumerate(rhymelist):
            if i == 0:  # first word, use P(w|x)
                myprob *= t_table[word_index, n]
            else:
                for word_index2 in rhymelist[:i]:  # history
                    myprob *= t_table[word_index, word_index2]
    if myprob == 0 and len(stanza) > 30:  # probably underflow
        myprob = 1e-300
    return myprob


def e_unnorm_post(t_table, words, stanzas, schemes, rprobs):
    """
    Expectation step: Compute posterior probability of schemes of appropriate length for each stanza
    """
    probs = numpy.zeros((len(stanzas), schemes.num_schemes))
    for i, stanza in enumerate(stanzas):
        scheme_indices = schemes.get_schemes_for_len(len(stanza))
        for scheme_index in scheme_indices:
            scheme = schemes.scheme_list[scheme_index]
            probs[i, scheme_index] = post_prob_scheme(t_table, words, stanza, scheme)
    probs = numpy.dot(probs, numpy.diag(rprobs))
    return probs


def e_norm_post(probs):
    """
    Normalize posterior probabilities
    """
    scheme_sums = numpy.sum(probs, axis=1)
    for i, scheme_sum in enumerate(list(scheme_sums)):
        if scheme_sum > 0:
            probs[i, :] /= scheme_sum
    return probs


def maximization_step(words, stanzas, schemes, probs):
    n = len(words)
    t_table = numpy.zeros((n, n + 1))
    rprobs = numpy.ones(schemes.num_schemes)
    for i, stanza in enumerate(stanzas):
        scheme_indices = schemes.get_schemes_for_len(len(stanza))
        for scheme_index in scheme_indices:
            myprob = probs[i, scheme_index]
            rprobs[scheme_index] += myprob
            scheme = schemes.scheme_list[scheme_index]
            rhymelists = get_rhymelists(stanza, scheme)
            for rhymelist in rhymelists:
                for j, word_index in enumerate(rhymelist):
                    t_table[word_index, n] += myprob
                    for word_index2 in rhymelist[:j] + rhymelist[j + 1:]:
                        t_table[word_index, word_index2] += myprob

    # Normalize t_table
    t_table_sums = numpy.sum(t_table, axis=0)
    for i, t_table_sum in enumerate(t_table_sums.tolist()):
        if t_table_sum != 0:
            t_table[:, i] /= t_table_sum

    # Normalize rprobs
    totrprob = numpy.sum(rprobs)
    rprobs /= totrprob
    return t_table, rprobs


def iterate(t_table, words, stanzas, schemes, rprobs, maxsteps):
    """
    Iterate steps 2-5 until convergence, return final probabilities
    """
    data_probs = numpy.zeros(len(stanzas))

    probs = None
    ctr = 0
    old_data_probs = None
    for ctr in range(maxsteps):
        old_data_probs = data_probs

        logging.info("Expectation step")
        probs = e_unnorm_post(t_table, words, stanzas, schemes, rprobs)

        # Estimate total probability
        data_probs = numpy.logaddexp.reduce(probs, axis=1)

        probs = e_norm_post(probs)  # normalize

        logging.info("Maximization step")
        t_table, rprobs = maximization_step(words, stanzas, schemes, probs)

        # Check convergence
        # if ctr > 0 and numpy.allclose(data_probs, old_data_probs):
        #     break

        logging.info("Iteration {}".format(ctr))

    # Error if it didn't converge
    if ctr == maxsteps - 1 and not numpy.allclose(data_probs, old_data_probs):
        logging.warning("Warning: EM did not converge")

    logging.info("Stopped after {} interations".format(ctr))
    return probs


def init_uniform_r(schemes):
    """assign equal prob to every scheme"""
    return numpy.ones(schemes.num_schemes) / schemes.num_schemes


def get_results(probs, stanzas, schemes):
    results = []
    for i, stanza in enumerate(stanzas):
        best_scheme = schemes.scheme_list[numpy.argmax(probs[i, :])]
        results.append((stanza.words, best_scheme))
    return results


def print_results(results, outfile):
    """write rhyme schemes at convergence"""
    for stanza_words, scheme in results:
        # scheme with highest probability
        outfile.write(str(' ').join(stanza_words) + str('\n'))
        outfile.write(str(' ').join(map(str, scheme)) + str('\n\n'))
    outfile.close()


def find_schemes(stanzas, t_table_init_function=init_uniform_ttable, num_iterations=10):
    scheme_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'allschemes.json')
    with open(scheme_filename, 'r') as scheme_file:
        schemes = Schemes(scheme_file)
    logging.info("Loaded files")

    words = get_wordlist(stanzas)
    for stanza in stanzas:
        stanza.set_word_indices(words)
    logging.info("Initialized {} words".format(len(words)))

    # Initialize p(r)
    rprobs = init_uniform_r(schemes)
    t_table = t_table_init_function(words)
    logging.info("Created t_table with shape {}".format(t_table.shape))
    final_probs = iterate(t_table, words, stanzas, schemes, rprobs, num_iterations)

    logging.info("EM done; writing results")
    results = get_results(final_probs, stanzas, schemes)
    return results


def main(args_list):
    parser = argparse.ArgumentParser(description='Discover schemes of given stanza file')
    parser.add_argument('infile', type=argparse.FileType('r'))
    parser.add_argument('init_type', choices=('u', 'o', 'p', 'd'), default='u')
    parser.add_argument('outfile', type=argparse.FileType('w'))
    parser.add_argument('-i, --iterations', dest='num_iterations', help='Number of iterations', type=int, default=100)
    parser.add_argument(
        '-v', '--verbose',
        help="Verbose output",
        action="store_const", dest="loglevel", const=logging.INFO,
    )
    args = parser.parse_args(args_list)
    logging.basicConfig(level=args.loglevel)

    stanzas = load_stanzas(args.infile)

    init_function = None
    if args.init_type == 'u':  # uniform init
        init_function = init_uniform_ttable
    elif args.init_type == 'o':  # init based on orthographic word sim
        init_function = init_basicortho_ttable
    elif args.init_type == 'p':  # init based on rhyming definition
        init_function = celex.init_perfect_ttable
    elif args.init_type == 'd':  # init based on difflib ratio
        init_function = init_difflib_ttable

    results = find_schemes(stanzas, init_function, args.num_iterations)

    print_results(results, args.outfile)
    logging.info("Wrote result")


if __name__ == '__main__':
    main(sys.argv[1:])