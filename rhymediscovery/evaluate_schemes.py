#!/usr/bin/env python

"""
Evaluate rhyme schemes against gold standard.
"""

from __future__ import division, print_function, unicode_literals

import argparse
import json
import os
import sys
from collections import defaultdict
from functools import reduce


class SuccessMeasure:

    def __init__(self):
        self.accuracy = None
        self.precision = None
        self.recall = None
        self.f_score = None

    def __str__(self):
        return """\
Accuracy:\t{accuracy:.4f}
Precision:\t{precision:.4f}
Recall:\t\t{recall:.4f}
F-score:\t{f_score:.4f}""".format(
            accuracy=self.accuracy,
            precision=self.precision,
            recall=self.recall,
            f_score=self.f_score,
        )


class EvaluationResult:

    def __init__(self):
        self.num_stanzas = None
        self.num_lines = None
        self.num_end_word_types = None
        self.naive_baseline_success = None
        self.less_naive_baseline_success = None
        self.input_success = None

    def __str__(self):
        s = """\
Num of stanzas: {num_stanzas}
Num of lines: {num_lines}
Num of end word types: {num_end_word_types}

Naive baseline:
{naive_baseline_success}

Less naive baseline:
{less_naive_baseline_success}
""".format(
            num_stanzas=self.num_stanzas,
            num_lines=self.num_lines,
            num_end_word_types=self.num_end_word_types,
            naive_baseline_success=self.naive_baseline_success,
            less_naive_baseline_success=self.less_naive_baseline_success,
        )
        if self.input_success:
            s += """
Input:
{input_success}
            """.format(
                input_success=self.input_success,
            )
        return s


def get_wordset(poems):
    """get all words"""
    words = sorted(list(set(reduce(lambda x, y: x + y, poems))))
    return words


def load_gold(gold_file):
    lines = gold_file.readlines()
    stanzas = []
    stanzaschemes = []
    for i, line in enumerate(lines):
        line = line.split()
        if i % 4 == 0:
            stanzas.append(tuple(line[1:]))
        elif i % 4 == 1:
            stanzaschemes.append(tuple(int(i) for i in line))
    return [stanzaschemes, stanzas]


def load_hypothesis(result_lines):
    stanzas = []
    schemes = []
    for i, line in enumerate(result_lines):
        line = line.split()
        if i % 3 == 0:
            stanzas.append(tuple(line))
        elif i % 3 == 1:
            schemes.append(tuple(int(i) for i in line))
    return zip(stanzas, schemes)


def compare(stanzas, gold_schemes, found_schemes):
    """get accuracy and precision/recall"""
    result = SuccessMeasure()
    total = float(len(gold_schemes))
    correct = 0.0
    for (g, f) in zip(gold_schemes, found_schemes):
        if g == f:
            correct += 1
    result.accuracy = correct / total

    # for each word, let rhymeset[word] = set of words in rest of stanza rhyming with the word
    # precision = # correct words in rhymeset[word]/# words in proposed rhymeset[word]
    # recall = # correct words in rhymeset[word]/# words in reference words in rhymeset[word]
    # total precision and recall = avg over all words over all stanzas

    tot_p = 0.0
    tot_r = 0.0
    tot_words = 0.0

    for (s, g, f) in zip(stanzas, gold_schemes, found_schemes):
        stanzasize = len(s)
        for wi, word in enumerate(s):
            grhymeset_word = set(
                map(lambda x: x[0], filter(lambda x: x[1] == g[wi], zip(range(wi + 1, stanzasize), g[wi + 1:]))))
            frhymeset_word = set(
                map(lambda x: x[0], filter(lambda x: x[1] == f[wi], zip(range(wi + 1, stanzasize), f[wi + 1:]))))

            if len(grhymeset_word) == 0:
                continue

            tot_words += 1

            if len(frhymeset_word) == 0:
                continue

            # find intersection
            correct = float(len(grhymeset_word.intersection(frhymeset_word)))
            precision = correct / len(frhymeset_word)
            recall = correct / len(grhymeset_word)
            tot_p += precision
            tot_r += recall

    precision = tot_p / tot_words
    recall = tot_r / tot_words
    result.precision = precision
    result.recall = recall
    if precision + recall > 0:
        result.f_score = 2 * precision * recall / (precision + recall)
    return result


def naive(gold_schemes):
    """find naive baseline (most common scheme of a given length)?"""
    scheme_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', 'schemes.json')
    with open(scheme_path, 'r') as f:
        dist = json.loads(f.read())
    best_schemes = {}
    for i in dist.keys():
        if not dist[i]:
            continue
        best_schemes[int(i)] = tuple(int(j) for j in (max(dist[i], key=lambda x: x[1])[0]).split())

    naive_schemes = []
    for g in gold_schemes:
        naive_schemes.append(best_schemes[len(g)])
    return naive_schemes


def less_naive(gold_schemes):
    """find 'less naive' baseline (most common scheme of a given length in subcorpus)"""
    best_schemes = defaultdict(lambda: defaultdict(int))
    for g in gold_schemes:
        best_schemes[len(g)][tuple(g)] += 1

    for i in best_schemes:
        best_schemes[i] = tuple(max(best_schemes[i].items(), key=lambda x: x[1])[0])

    naive_schemes = []
    for g in gold_schemes:
        naive_schemes.append(best_schemes[len(g)])
    return naive_schemes


def evaluate(gstanzaschemes, gstanzas, hypothesis):
    words = get_wordset(gstanzas)

    result = EvaluationResult()
    result.num_stanzas = len(gstanzas)
    result.num_lines = sum(map(len, gstanzas))
    result.num_end_word_types = len(words)

    naive_schemes = naive(gstanzaschemes)
    result.naive_baseline_success = compare(gstanzas, gstanzaschemes, naive_schemes)

    less_naive_schemes = less_naive(gstanzaschemes)
    result.less_naive_baseline_success = compare(gstanzas, gstanzaschemes, less_naive_schemes)

    hstanzaschemes = [scheme for (stanza, scheme) in hypothesis]
    result.input_success = compare(gstanzas, gstanzaschemes, hstanzaschemes)
    return result


def main(args_list=None):
    args_list = args_list or sys.argv[1:]
    parser = argparse.ArgumentParser(description='Evaluate find_schemes output')
    parser.add_argument(
        'infile',
        help='Output of find_schemes that is evaluated',
        nargs='?',
        type=argparse.FileType('r'),
        default=sys.stdin,
    )
    parser.add_argument(
        'gold_file',
        help='Stanzas annotated with correct rhyme schemes',
        type=argparse.FileType('r'),
    )
    args = parser.parse_args(args_list)

    gstanzaschemes, gstanzas = load_gold(args.gold_file)
    args.gold_file.close()

    hypothesis_lines = args.infile.readlines()
    hstanzaschemes = load_hypothesis(hypothesis_lines)

    result = evaluate(gstanzaschemes, gstanzas, hstanzaschemes)
    print(result)


if __name__ == '__main__':
    main()
