#!/usr/bin/env python

"""
EM algorithm for learning rhyming words and rhyme schemes with independent stanzas.
Original implementation: Sravana Reddy (sravana@cs.uchicago.edu), 2011.
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

from rhymediscovery import celex


def load_stanzas(stanzas_file):
    """
    Load stanzas from gold standard file
    """
    f = stanzas_file.readlines()
    stanzas = []
    for i, line in enumerate(f):
        if i % 4 == 0:
            stanza_words = line.strip().split()[1:]
            stanzas.append(Stanza(stanza_words))
    return stanzas


class Stanza:

    def __init__(self, stanza_words):
        self.words = tuple(stanza_words)  # Sequence of final words
        self.word_indices = None  # Indices of words with respect to global wordlist

    def set_word_indices(self, wordlist):
        """
        Populate the list of word_indices, mapping self.words to the given wordlist
        """
        self.word_indices = [wordlist.index(word) for word in self.words]

    def __str__(self):
        return ' '.join(self.words)

    def __len__(self):
        return len(self.words)


class Schemes:
    """
    Stores schemes loaded from schemes.json
    """

    def __init__(self, scheme_file):
        self.scheme_file = scheme_file
        self.scheme_list, self.scheme_dict = self._parse_scheme_file()
        self.num_schemes = len(self.scheme_list)

    def _parse_scheme_file(self):
        """
        Initialize redundant data structures for lookup optimization
        """
        schemes = json.loads(self.scheme_file.read(), object_pairs_hook=OrderedDict)
        scheme_list = []
        scheme_dict = defaultdict(list)
        for scheme_len, scheme_group in schemes.items():
            for scheme_str, _count in scheme_group:
                scheme_code = tuple(int(c) for c in scheme_str.split(' '))
                scheme_list.append(scheme_code)
                scheme_dict[int(scheme_len)].append(len(scheme_list) - 1)
        return scheme_list, scheme_dict

    def get_schemes_for_len(self, n):
        """
        Returns the indices of all n-length schemes in self.scheme_list
        """
        return self.scheme_dict[n]


def get_wordlist(stanzas):
    """
    Get an iterable of all final words in all stanzas
    """
    return sorted(list(set().union(*[stanza.words for stanza in stanzas])))


def get_rhymelists(stanza, scheme):
    """
    Returns ordered lists of the stanza's word indices as defined by given scheme
    """
    rhymelists = defaultdict(list)
    for rhyme_group, word_index in zip(scheme, stanza.word_indices):
        rhymelists[rhyme_group].append(word_index)
    return list(rhymelists.values())


def init_distance_ttable(wordlist, distance_function):
    """
    Initialize pair-wise rhyme strenghts according to the given word distance function
    """
    n = len(wordlist)
    t_table = numpy.zeros((n, n + 1))

    # Initialize P(c|r) accordingly
    for r, w in enumerate(wordlist):
        for c, v in enumerate(wordlist):
            if c < r:
                t_table[r, c] = t_table[c, r]  # Similarity is symmetric
            else:
                t_table[r, c] = distance_function(w, v) + 0.001  # For backoff
    t_table[:, n] = numpy.mean(t_table[:, :-1], axis=1)

    # Normalize
    t_totals = numpy.sum(t_table, axis=0)
    for i, t_total in enumerate(t_totals.tolist()):
        t_table[:, i] /= t_total
    return t_table


def init_uniform_ttable(wordlist):
    """
    Initialize (normalized) theta uniformly
    """
    n = len(wordlist)
    return numpy.ones((n, n + 1)) * (1 / n)


def basic_word_sim(word1, word2):
    """
    Simple measure of similarity: Number of letters in common / max length
    """
    return sum([1 for c in word1 if c in word2]) / max(len(word1), len(word2))


def init_basicortho_ttable(wordlist):
    return init_distance_ttable(wordlist, basic_word_sim)


def difflib_similarity(word1, word2):
    """
    Distance function using the built-in difflib ratio
    """
    sequence_matcher = SequenceMatcher(None, word1, word2)
    return sequence_matcher.ratio()


def init_difflib_ttable(wordlist):
    return init_distance_ttable(wordlist, difflib_similarity)


def post_prob_scheme(t_table, stanza, scheme):
    """
    Compute posterior probability of a scheme for a stanza, with probability of every word in rhymelist
    rhyming with all the ones before it
    """
    myprob = 1
    rhymelists = get_rhymelists(stanza, scheme)
    for rhymelist in rhymelists:
        for i, word_index in enumerate(rhymelist):
            if i == 0:  # first word, use P(w|x)
                myprob *= t_table[word_index, -1]
            else:
                for word_index2 in rhymelist[:i]:  # history
                    myprob *= t_table[word_index, word_index2]
    if myprob == 0 and len(stanza) > 30:  # probably underflow
        myprob = 1e-300
    return myprob


def expectation_step(t_table, stanzas, schemes, rprobs):
    """
     Compute posterior probability of schemes for each stanza
    """
    probs = numpy.zeros((len(stanzas), schemes.num_schemes))
    for i, stanza in enumerate(stanzas):
        scheme_indices = schemes.get_schemes_for_len(len(stanza))
        for scheme_index in scheme_indices:
            scheme = schemes.scheme_list[scheme_index]
            probs[i, scheme_index] = post_prob_scheme(t_table, stanza, scheme)
    probs = numpy.dot(probs, numpy.diag(rprobs))

    # Normalize
    scheme_sums = numpy.sum(probs, axis=1)
    for i, scheme_sum in enumerate(scheme_sums.tolist()):
        if scheme_sum > 0:
            probs[i, :] /= scheme_sum
    return probs


def maximization_step(num_words, stanzas, schemes, probs):
    """
    Update latent variables t_table, rprobs
    """
    t_table = numpy.zeros((num_words, num_words + 1))
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
                    t_table[word_index, -1] += myprob
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


def iterate(t_table, wordlist, stanzas, schemes, rprobs, maxsteps):
    """
    Iterate EM and return final probabilities
    """
    data_probs = numpy.zeros(len(stanzas))
    old_data_probs = None
    probs = None
    num_words = len(wordlist)

    ctr = 0
    for ctr in range(maxsteps):
        logging.info("Iteration {}".format(ctr))
        old_data_probs = data_probs

        logging.info("Expectation step")
        probs = expectation_step(t_table, stanzas, schemes, rprobs)

        logging.info("Maximization step")
        t_table, rprobs = maximization_step(num_words, stanzas, schemes, probs)

    # Warn if it did not converge
    data_probs = numpy.logaddexp.reduce(probs, axis=1)
    if ctr == maxsteps - 1 and not numpy.allclose(data_probs, old_data_probs):
        logging.warning("Warning: EM did not converge")

    logging.info("Stopped after {} interations".format(ctr))
    return probs


def init_uniform_r(schemes):
    """
    Assign equal probability to all schemes
    """
    return numpy.ones(schemes.num_schemes) / schemes.num_schemes


def get_results(probs, stanzas, schemes):
    """
    Returns a list of tuples (
        stanza [as list of final words],
        best scheme [as list of integers]
    )
    """
    results = []
    for i, stanza in enumerate(stanzas):
        best_scheme = schemes.scheme_list[numpy.argmax(probs[i, :])]
        results.append((stanza.words, best_scheme))
    return results


def print_results(results, outfile):
    """
    Write results to outfile
    """
    for stanza_words, scheme in results:
        outfile.write(str(' ').join(stanza_words) + str('\n'))
        outfile.write(str(' ').join(map(str, scheme)) + str('\n\n'))
    outfile.close()
    logging.info("Wrote result")


def find_schemes(stanzas, t_table_init_function=init_uniform_ttable, num_iterations=10):
    # Allow input of string lists as stanzas
    if not isinstance(stanzas[0], Stanza):
        stanzas = [Stanza(words) for words in stanzas]

    scheme_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', 'schemes.json')
    with open(scheme_filename, 'r') as scheme_file:
        schemes = Schemes(scheme_file)
    logging.info("Loaded files")

    wordlist = get_wordlist(stanzas)
    for stanza in stanzas:
        stanza.set_word_indices(wordlist)
    logging.info("Initialized list of {} words".format(len(wordlist)))

    # Initialize p(r)
    rprobs = init_uniform_r(schemes)
    t_table = t_table_init_function(wordlist)
    logging.info("Created t_table with shape {}".format(t_table.shape))
    final_probs = iterate(t_table, wordlist, stanzas, schemes, rprobs, num_iterations)

    logging.info("EM done; writing results")
    results = get_results(final_probs, stanzas, schemes)
    return results


def main(args_list=None):
    """
    Wrapper for find_schemes if called from command line
    """
    args_list = args_list or sys.argv[1:]
    parser = argparse.ArgumentParser(description='Discover schemes of given stanza file')
    parser.add_argument(
        'infile',
        type=argparse.FileType('r'),
    )
    parser.add_argument(
        'outfile',
        help='Where the result is written to. Default: stdout',
        nargs='?',
        type=argparse.FileType('w'),
        default=sys.stdout,
    )
    parser.add_argument(
        '-t --init-type',
        help='Whether to initialize theta uniformly (u), with the orthographic similarity '
             'measure (o), or using CELEX pronunciations and definition of rhyme (p). '
             'The last one requires you to have CELEX on your machine.',
        dest='init_type',
        choices=('u', 'o', 'p', 'd'),
        default='o',
    )
    parser.add_argument(
        '-i, --iterations',
        help='Number of iterations (default: 10)',
        dest='num_iterations',
        type=int,
        default=10,
    )
    parser.add_argument(
        '-v', '--verbose',
        help="Verbose output",
        action="store_const",
        dest="loglevel",
        const=logging.INFO,
    )
    args = parser.parse_args(args_list)
    logging.basicConfig(level=args.loglevel)

    stanzas = load_stanzas(args.infile)

    init_function = None
    if args.init_type == 'u':
        init_function = init_uniform_ttable
    elif args.init_type == 'o':
        init_function = init_basicortho_ttable
    elif args.init_type == 'p':
        init_function = celex.init_perfect_ttable
    elif args.init_type == 'd':
        init_function = init_difflib_ttable

    results = find_schemes(stanzas, init_function, args.num_iterations)

    print_results(results, args.outfile)


if __name__ == '__main__':
    main()
