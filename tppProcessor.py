#!/usr/bin/python3

import re
import spacy
import sys
import logging
from spacy.attrs import ORTH, DEP, POS, LEMMA, HEAD
from spacy.pipeline import EntityRecognizer
from text_sections import *
from collections import Counter


TEST_FILE = "data/tpp_fragment.txt"

SECTION_PAT = r'^Section ([A-Z]): (.+)$'
ARTICLE_PAT = r'^Article (\d+\.\d+): (.+)$'
ANNEX_PAT = r'^Annex ((\d-)?[A-Z]): (.*)$'
CHAPTER_PAT = r'^Chapter (\d+) – (.*)$'

CHAPTER = re.compile(CHAPTER_PAT, re.MULTILINE)
SECTION = re.compile(SECTION_PAT, re.MULTILINE)
ARTICLE = re.compile(ARTICLE_PAT, re.MULTILINE)
ANNEX = re.compile(ANNEX_PAT, re.MULTILINE)

class Paragraph(PassageUnit):

    pattern = NUM_PGPH
    SubUnit = None

    def __init__(self, header, text):
        super().__init__(header, text)


class Article(IndexUnit):

    pattern = ARTICLE
    SubUnit = Paragraph

    def __init__(self, header, text):
        super().__init__(header, text)

class Section(IndexUnit):

    SubUnit = Article
    pattern = SECTION

    def __init__(self, header, text):
        super().__init__(header, text)


class Chapter(IndexUnit):

    SubUnit = Section
    pattern = CHAPTER

    def __init__(self, header, text):
        super().__init__(header, text)


class TppFull(IndexUnit):

    SubUnit = Chapter

    def __init__(self, text):
        super().__init__("TPP", text)


# TEST CODE

with open(TEST_FILE) as tppfile:
    tpp = tppfile.read()

print("\nprocessing TPP text")
proc = TppFull(tpp)

print("\n\n")
print("Success!")
print("\n\n\n")

chapter = proc.chapter['1']
sec = chapter.section['A']
art = sec.article['1.2']
par = art.paragraph['1']

root_hist = Counter()
vb_hist = Counter()
np_hist = Counter()

print("\ngetting a few histograms..")
for sent in proc.getFlatText():
	nps = set(sent.noun_chunks)
	np_hist.update(nps)
	for tok in sent:
		lem = tok.lemma_
		pos = tok.pos_
		dep = tok.dep_
		if dep == "ROOT":
			root_hist.update([lem])
		if pos == "VERB":
			vb_hist.update([lem])
			

print("DONE!")