#!/usr/bin/env python

"""EM algorithm for learning rhyming words and rhyme schemes with independent stanzas.
Sravana Reddy (sravana@cs.uchicago.edu), 2011.
"""

from __future__ import division, print_function, unicode_literals

import argparse
import json
import logging
import os
import random
import sys
from collections import defaultdict, OrderedDict
from functools import reduce

import numpy

import celex


def load_stanzas(stanzas_file):
    """Load raw stanzas from gold standard file"""
    f = stanzas_file.readlines()
    stanzas = []
    for i, line in enumerate(f):
        line = line.split()
        if i % 4 == 0:
            stanzas.append(line[1:])
    return stanzas


class Schemes:

    def __init__(self, scheme_filename):
        self.scheme_filename = scheme_filename
        self.scheme_list = self._parse_scheme_file()
        self.num_schemes = len(self.scheme_list)
        self.scheme_array = self._create_scheme_array()

    def _parse_scheme_file(self):
        with open(self.scheme_filename, 'r') as f:
            schemes = json.loads(f.read(), object_pairs_hook=OrderedDict)
        scheme_list = []
        for scheme_group in schemes.values():
            for scheme_str, count in scheme_group:
                scheme_code = [int(c) for c in scheme_str.split(' ')]
                scheme_list.append(scheme_code)
        return scheme_list

    def _create_scheme_array(self):
        return numpy.arange(self.num_schemes)

    def get_schemes_for_len(self, n):
        """
        :return: List of indices of schemes with length n
        """
        return [i for i, scheme in enumerate(self.scheme_list) if len(scheme) == n]


def get_wordset(stanzas):
    """get all words"""
    words = sorted(list(set(reduce(lambda x, y: x + y, stanzas))))
    return words


def get_rhymelists(stanza, scheme):
    """
    Transform stanza into list of ordered lists of rhymesets as defined by rhyme scheme
    """
    rhymelists = defaultdict(list)
    for stanzaword, schemeword in zip(stanza, scheme):
        rhymelists[schemeword].append(stanzaword)
    return list(rhymelists.values())


def init_uniform_ttable(words):
    """initialize (normalized) theta uniformly"""
    n = len(words)
    return numpy.ones((n, n + 1)) * (1 / n)


def basic_word_sim(word1, word2):
    """Simple measure of similarity: num of letters in common/max length"""
    common = 0.0
    if word1 == word2:
        return 1.0
    for c in word1:
        if c in word2:
            common += 1
    return common / max(len(word1), len(word2))


def init_basicortho_ttable(words):
    """initialize probs according to simple measure of orthographic similarity"""
    n = len(words)
    t_table = numpy.zeros((n, n + 1))

    # initialize P(c|r) accordingly
    for r, w in enumerate(words):
        for c, v in enumerate(words):
            if c < r:
                t_table[r, c] = t_table[c, r]  # similarity is symmetric
            else:
                t_table[r, c] = basic_word_sim(w, v) + 0.001  # for backoff
        t_table[r, n] = random.random()  # no estimate for P(r|no history)

    # normalize
    for c in range(n + 1):
        tot = sum(t_table[:, c])
        for r in range(n):
            t_table[r, c] = t_table[r, c] / tot

    return t_table


def post_prob_scheme(t_table, words, stanza, scheme):
    """
    Compute posterior probability of a scheme for a stanza, with probability of every word in rhymelist
    rhyming with all the ones before it
    """
    myprob = 1
    n = len(words)
    rhymelists = get_rhymelists(stanza, scheme)
    for rhymelist in rhymelists:
        for i, w in enumerate(rhymelist):
            r = words.index(w)
            if i == 0:  # first word, use P(w|x)
                myprob = t_table[r, n]
            else:
                for v in rhymelist[:i]:  # history
                    c = words.index(v)
                    myprob *= t_table[r, c]
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
            # TODO Multiply with rprobs column-wise
            posterior_prob = rprobs[scheme_index] * post_prob_scheme(t_table, words, stanza, scheme)
            probs[i, scheme_index] = posterior_prob
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
                for j, w in enumerate(rhymelist):
                    r = words.index(w)
                    t_table[r, n] += myprob
                    for v in rhymelist[:j] + rhymelist[j + 1:]:
                        c = words.index(v)
                        t_table[r, c] += myprob

    # Normalize t_table
    for c in range(n + 1):
        tot = numpy.sum(t_table[:, c])
        if tot == 0:
            continue
        t_table[:, c] /= tot

    # Normalize rprobs
    totrprob = numpy.sum(rprobs)
    rprobs /= totrprob
    return t_table, rprobs


def iterate(t_table, words, stanzas, schemes, rprobs, maxsteps):
    """iterate steps 2-5 until convergence, return final t_table"""
    data_prob = -10 ** 10
    epsilon = 0.1

    probs = None
    ctr = 0
    old_data_prob = None
    for ctr in range(maxsteps):
        old_data_prob = data_prob

        # E-step
        probs = e_unnorm_post(t_table, words, stanzas, schemes, rprobs)

        # estimate total probability
        allschemeprobs = numpy.sum(probs, axis=1)

        if 0.0 in allschemeprobs:
            # This may happen for very large data on large stanzas, small hack to prevent
            underflows = filter(
                lambda x: x[2] == 0.0,
                zip(range(len(stanzas)), stanzas, allschemeprobs)
            )
            for underflow in underflows:
                if len(probs[underflow[0]]) == 1:
                    probs[underflow[0]][0] = 1e-300
                    allschemeprobs[underflow[0]] = 1e-300
                    logging.warning("Fixed underflow error on {}".format(underflow[1]))
                else:
                    logging.warning("Problem! {} {}".format(underflow, probs[underflow[0]]))

        allschemeprobs = numpy.log2(allschemeprobs)

        probs = e_norm_post(probs)  # normalize

        # M-step
        t_table, rprobs = maximization_step(words, stanzas, schemes, probs)

        # check convergence
        data_prob = numpy.sum(allschemeprobs)
        if ctr > 0 and data_prob - old_data_prob < epsilon:
            break

        logging.info("Iteration {} -- Log likelihood of data: {}".format(ctr, data_prob))

    # error if it didn't converge
    if ctr == maxsteps - 1 and data_prob - old_data_prob >= epsilon:
        logging.warning("Warning: EM did not converge")

    return probs, data_prob


def init_uniform_r(schemes):
    """assign equal prob to every scheme"""
    return numpy.ones(schemes.num_schemes) / schemes.num_schemes


def get_results(probs, stanzas, schemes):
    results = []
    for i, stanza in enumerate(stanzas):
        best_scheme = schemes.scheme_list[numpy.argmax(probs[i, :])]
        results.append((stanza, best_scheme))
    return results


def print_results(results, outfile):
    """write rhyme schemes at convergence"""
    for stanza, scheme in results:
        # scheme with highest probability
        outfile.write(str(' ').join(stanza) + str('\n'))
        outfile.write(str(' ').join(map(str, scheme)) + str('\n\n'))
    outfile.close()


def find_schemes(stanzas, t_table_init_function=init_uniform_ttable):
    scheme_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'allschemes.json')
    schemes = Schemes(scheme_filename)
    logging.info("Loaded files")
    words = get_wordset(stanzas)
    # Initialize p(r)
    rprobs = init_uniform_r(schemes)
    t_table = t_table_init_function(words)
    logging.info("Initialized {} words".format(len(words)))
    final_probs, data_prob = iterate(t_table, words, stanzas, schemes, rprobs, 100)

    results = get_results(final_probs, stanzas, schemes)
    return results


def main(args_list):
    parser = argparse.ArgumentParser(description='Discover schemes of given stanza file')
    parser.add_argument('infile', type=argparse.FileType('r'))
    parser.add_argument('init_type', choices=('u', 'o', 'p'), default='u')
    parser.add_argument('outfile', type=argparse.FileType('w'))
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

    results = find_schemes(stanzas, init_function)

    print_results(results, args.outfile)
    logging.info("Wrote result")


if __name__ == '__main__':
    main(sys.argv[1:])
