from __future__ import unicode_literals

import random
import re
from collections import defaultdict

import numpy

VOWELS_RE = re.compile('[iye|aou#$(IYE/A{&QO}VU@!)*<cq0~^KLM123456789WBX]')
CELEX_DIR = '../../data/celex/CELEX_V2/'  # change to the location of your CELEX directory
EPW_FILE = CELEX_DIR + '/english/epw/epw.cd'


def read_celex():
    spam = map(lambda x: x.strip().split('\\'), open(EPW_FILE).readlines())
    spam = map(lambda x: (x[1], x[6].replace('-', '').replace('"', "'")), spam)
    d = defaultdict(list)
    for (word, pron) in spam:
        if "'" in pron:  # can only test words with at least on stressed syllable
            d[word].append(pron)
    return d


def is_rhyme(d, w1, w2):
    """check if words rhyme"""
    for p1 in d[w1]:
        # extract only "rhyming portion"
        p1 = p1.split("'")[-1]
        m = VOWELS_RE.search(p1)
        if not m:
            print(p1)
        p1 = p1[m.start():]
        for p2 in d[w2]:
            p2 = p2.split("'")[-1]
            m = VOWELS_RE.search(p2)
            if not m:
                print(w2, p2)
            p2 = p2[m.start():]
            if p1 == p2:
                return True
    return False


def init_perfect_ttable(words):
    """initialize (normalized) theta according to whether words rhyme"""
    d = read_celex()

    not_in_dict = 0

    n = len(words)
    t_table = numpy.zeros((n, n + 1))

    # initialize P(c|r) accordingly
    for r, w in enumerate(words):
        if w not in d:
            not_in_dict += 1
        for c, v in enumerate(words):
            if c < r:
                t_table[r, c] = t_table[c, r]
            elif w in d and v in d:
                t_table[r, c] = int(is_rhyme(d, w, v)) + 0.001  # for backoff
            else:
                t_table[r, c] = random.random()
        t_table[r, n] = random.random()  # no estimate for P(r|no history)

    print(not_in_dict, "of", n, " words are not in CELEX")

    # normalize
    for c in range(n + 1):
        tot = sum(t_table[:, c])
        for r in range(n):
            t_table[r, c] = t_table[r, c] / tot

    return t_table
