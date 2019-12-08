#!/usr/bin/python3

import re
import spacy
import sys
import logging
from spacy.attrs import ORTH, DEP, POS, LEMMA, HEAD
from spacy.pipeline import EntityRecognizer
from text_sections import *


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
        self.entities = set()
        self.conditions = []

    def _getEntities(self):
        raise NotImplemented


class Section(IndexUnit):

    SubUnit = Article
    pattern = SECTION

    def __init__(self, header, text):
        super().__init__(header, text)
        # print("finished processing section: {}\n".format(header))


class Chapter(IndexUnit):

    SubUnit = Section
    pattern = CHAPTER

    def __init__(self, header, text):
        print("let's check this out")
        super().__init__(header, text)


class TppFull(IndexUnit):

    SubUnit = Chapter

    def __init__(self, text):
        print(super())
        IndexUnit.__init__(self, "TPP", text)



# if __name__ == "__main__":
with open(TEST_FILE) as tppfile:
    tpp = tppfile.read()


proc = TppFull(tpp)
print("\n\n")
print("Success!")
print("\n\n\n")


# art = sec.article['2.3']
    # print(art.header, "\n", art.doc)
