#!/usr/bin/env python

"""Evaluate rhyme schemes against gold standard.
Also contains some utilities to parse data.
Jan 2011."""

from __future__ import print_function, unicode_literals

import argparse
import json
import os
import sys
from collections import defaultdict
from functools import reduce


def get_wordset(poems):
    """get all words"""
    words = sorted(list(set(reduce(lambda x, y: x + y, poems))))
    return words


def load_gold(gold_file):
    lines = gold_file.readlines()
    stanzas = []
    stanzaschemes = []
    poemschemes = []
    for i, line in enumerate(lines):
        line = line.split()
        if i % 4 == 0:
            stanzas.append(line[1:])
        elif i % 4 == 1:
            if not line:
                print("Error in gold!", i, lines[i - 1], lines[i - 2])
            stanzaschemes.append(line)
        elif i % 4 == 2:
            poemschemes.append(line)
    return [stanzaschemes, poemschemes, stanzas]


def load_result(result_lines):
    stanzas = []
    schemes = []
    for i, line in enumerate(result_lines):
        line = line.split()
        if i % 3 == 0:
            stanzas.append(line[1:])
        elif i % 3 == 1:
            if not line:
                print("Error in result!", i, result_lines[i - 1], result_lines[i - 2])
            schemes.append(line)
    return [schemes, stanzas]


def compare(stanzas, gold_schemes, found_schemes):
    """get accuracy and precision/recall"""
    total = float(len(gold_schemes))
    correct = 0.0
    for (g, f) in zip(gold_schemes, found_schemes):
        if g == f:
            correct += 1
    print("Accuracy", correct, total, 100 * correct / total)

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
    print("Precision", precision)
    print("Recall", recall)
    print("F-score", 2 * precision * recall / (precision + recall))


def naive(gold_schemes):
    """find naive baseline (most common scheme of a given length)?"""
    scheme_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'allschemes.json')
    with open(scheme_path, 'r') as f:
        dist = json.loads(f.read())
    best_schemes = {}
    for i in dist:
        if not dist[i]:
            continue
        best_schemes[int(i)] = (max(dist[i], key=lambda x: x[1])[0]).split()

    naive_schemes = []
    for g in gold_schemes:
        naive_schemes.append(best_schemes[len(g)])
    return naive_schemes


def less_naive(gold_schemes):
    """find 'less naive' baseline (most common scheme of a given length in subcorpus)"""
    best_schemes = defaultdict(lambda: defaultdict(int))
    for g in gold_schemes:
        best_schemes[len(g)][tuple(g)] += 1

    m = sum(map(len, best_schemes.values()))

    for i in best_schemes:
        best_schemes[i] = list(max(best_schemes[i].items(), key=lambda x: x[1])[0])

    naive_schemes = []
    for g in gold_schemes:
        naive_schemes.append(best_schemes[len(g)])
    return naive_schemes


def main(args_list):
    parser = argparse.ArgumentParser(description='Evaluate findschemes output')
    parser.add_argument('gold_file', type=argparse.FileType('r'))
    parser.add_argument('hypothesis_file', nargs='?', type=argparse.FileType('r'), default=sys.stdin)
    args = parser.parse_args(args_list)

    [gstanzaschemes, gpoemschemes, gstanzas] = load_gold(args.gold_file)

    words = get_wordset(gstanzas)
    n = len(words)

    # for stanzas
    print('Num of stanzas: ', len(gstanzas))
    print('Num of lines: ', sum(map(len, gstanzas)))
    print('Num of end word types: ', len(words))
    print()

    naive_schemes = naive(gstanzaschemes)
    print("Naive baseline:")
    compare(gstanzas, gstanzaschemes, naive_schemes)
    print()

    less_naive_schemes = less_naive(gstanzaschemes)
    print("Less naive baseline:")
    compare(gstanzas, gstanzaschemes, less_naive_schemes)
    print()

    hypothesis_lines = args.hypothesis_file.readlines()
    if hypothesis_lines:
        [hstanzaschemes, hstanzas] = load_result(hypothesis_lines)
        print(args.hypothesis_file.name, ":")
        compare(gstanzas, gstanzaschemes, hstanzaschemes)
        print()


if __name__ == '__main__':
    main(sys.argv[1:])
