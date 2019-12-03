#!/usr/bin/python3

import re
import spacy
import sys
import logging
from spacy.attrs import ORTH, DEP, POS, LEMMA, HEAD
from spacy.pipeline import EntityRecognizer
from abc import ABC, abstractmethod
from collections import ChainMap


SPACY_MODEL = "en_core_web_md"
TEST_FILE = "data/tpp_fragment.txt"

# bullet point sequences
lalpha = "abcdefghijklmnopqrstuvwxyz"
ualpha = lalpha.upper()


# when we're parsing condition lists, etc.
# which may be nested, we need the bullet point
# formats (in the TPP's case, i, upper and lowercase
# alpha). Since i could also be a lowercase alpha
# we need to use the previous key to see if we're
# starting a lower level or resuming a higher one


# Declare and compile regex objects for identifying subunits
SECTION_PAT = r'^Section ([A-Z]): (.+)$'
ARTICLE_PAT = r'^Article (\d+\.\d+): (.+)$'
ANNEX_PAT = r'^Annex ((\d-)?[A-Z]): (.*)$'
SUBITEM_PAT = r'^(\([a-z]\)|\d)(.*)$'
SUBITEM_HEAD_PAT = r':$\s+\(.+\)'
CHAPTER_PAT = r'^Chapter (\d+) – (.*)$'
IPOINT_PAT = r'i+'
TYPE_MATCH = r"<class '__main__.(\w*)'>"
NUM_PGPH_PAT = r'^(\d)\.\s'
POINT_PAT = r'\((.+)\)'

IPOINT = re.compile(IPOINT)
POINT = re.compile(POINT_PAT, re.MULTLINE)
NUM_PGPH = re.compile(NUM_PGPH_PAT, re.MULTILINE)
TYPE_MATCHER = re.compile(TYPE_MATCH)
SUBITEM = re.compile(SUBITEM_PAT, re.MULTILINE)
SUBITEM_HEAD = re.compile(SUBITEM_HEAD_PAT, re.MULTILINE)
CHAPTER = re.compile(CHAPTER_PAT, re.MULTILINE)
SECTION = re.compile(SECTION_PAT, re.MULTILINE)
ARTICLE = re.compile(ARTICLE_PAT, re.MULTILINE)
ANNEX = re.compile(ANNEX_PAT, re.MULTILINE)

# Load spacy model
print("loading model")
nlp = spacy.load(SPACY_MODEL)
ner = EntityRecognizer(nlp.vocab)
print("\ncomplete\n")


# Helper functions for organizing
# logic implicit in formatting

def alph_is_prev(current, prev, seq=lalpha):
    idx = seq.index(current)
    test = seq[idx-1]
    return True if prev == test else False


def is_prev(current, prev):
    # 1st check if we've descended into i indexed constituents
    # TODO(this ignores edgecase where letter i
          # descends from alphabetical line . Not
          # included in current example text, but
          # once the flow is more crystallized,
          # this will need to be fixed
    if not IPOINT.match(current) or prev == 'h':
        return alph_is_prev(current, prev, seq)
    else:
        # current is at least one or more i in seq
        return True if current[:-1] == prev else False


class TextProcessor(ABC):

    @abstractmethod
    def process(self, raw):
        self._preprocess(raw)

    def _preprocess(self, raw):
        raise NotImplemented


class TextUnit(ABC):
    def __init__(self, header, text, SubUnit):
        self.header = header
        # if this is implementing a class
        # representing a terminal subunit,
        # we want container attributes for
        # nlp data rather than subunit data
        if self.SubUnit:
            # Name subunit attribute dynamically
            if not issubclass(self.SubUnit, TextUnit):
                bad_type = type(self.SubUnit)
                raise TypeError("Subunit is of type {}".format(bad_type))
            subunit_name = SubUnit.__name__.lower()
            subunits_or_not = self._getSubUnits(text)
            if subunits_or_not:
                setattr(self, subunit_name, subunits_or_not)
        # Since _getSubUnits could
        # reach terminal subunit and change the
        # value of self.SubUnit to None,
        # use if not, rather than else
        if not self.SubUnit:
            self._beTerminal(text)

    @abstractmethod
    def _getSubUnits(self, raw):
        raise NotImplemented

    def getSubUnitName(self):
        if self.SubUnit:
            return self.SubUnit.__name__.lower()
        else:
            return None

    def leafType(self):
        type_ptr = cls
        while type_ptr:
            type_ptr = type_ptr.SubUnit
        return type_ptr

    def _textsOrSubUnits(self):
        if not self.SubUnit:
            return [self.raw]
        else:
            out = []
            # Assume we have a subclass of TextUnit
            subdat = getattr(self,
                             self.SubUnit.__name__.lower())
            for sbu in subdat.values():
                if not sbu.SubUnit:
                    sbu_out = [obj.raw for obj in sbu]
                else:
                    sbu_out = sbu._textsOrSubUnits()
                out += sbu_out
            return out

    def _getSubHeaders(self, raw):
        if self.SubUnit:
            smatches = list(self.SubUnit.pattern.finditer(raw))
        else:
            smatches = []
        return smatches

    def _getTypeName(self):
        try:
            name = TYPE_MATCHER.search(str(type(self))).group(1)
            return name if name else None
        except AttributeError:
            return str(type(self))

    def _beTerminal(self, raw):
        if hasattr(self, 'subunits'):
            delattr(self, 'subunits')
        self.raw = raw
        # TODO: update this to user logging
        # for speed/bottleneck tests
        print("processing {} headed: {}".format(
            self._getTypeName(), self.header))
        self.text = nlp(raw)
        print("complete\n\n")
        self.entities = set()
        self.conditions = []


class IndexUnit(TextUnit):
    def __init__(self, header, text):
        super().__init__(header)

    def _getSubUnits(self, raw):
        if self.SubUnit is None:
            return None
        smatches = self._getSubHeaders(raw)
        while not smatches:
            # print(type(self))
            if self.SubUnit is not None:
                self.SubUnit = self.SubUnit.SubUnit
            else:
                # this feels sketchy
                self._beTerminal(raw)
                break
            smatches = self._getSubHeaders(raw)

        subunits = dict()
        while smatches:
            # section match matches
            # the bill code and title
            section_match = smatches.pop(0)
            scode = section_match.group(1)
            header = section_match.group(2)
            # if we're not at the end,
            # assume the end of the text
            # precedes the title of the next section
            if smatches:
                text_end = smatches[0].start() - 1
            else:
                text_end = len(raw) - 1
            text = raw[section_match.end():text_end].strip()
            subunits[scode] = self.SubUnit(header, text)
        return subunits


class PassageUnit(TextUnit):
    '''A terminal text unit, such as a paragraph.
    The idea is that chunks inherited from this class do not
    bear any direct syntactic relationship to their siblings
    in the tree'''

    # to process conditions implicit in formattting
    _bpoint_markers = {
        "CAP": re.compile(r'[A-Z]'),
        "LWR": re.compile(r'[a-z]'),
        "IDX": re.compile(r'[iI]+')
    }

    def __init__(self, text):
        super().__init__(header, text, None)
        self.content = {}

    def _getBpointType(self, point):
        for k in self._bpoint_markers:
            if self._bpoint_markers[k].match(point):
                return k

    def _getPredicates(self, span):
        '''span is the slice from the raw
        text that contains a serialized 
        set of predicates'''
        ptrace = list()
        tag = "INIT"
        predicates = []
        text = self.raw[span]
        items = SUBITEM.finditer(text)
        while items:
            pt = next(items)
            this_tag = self._getBpointType(pt.group(1))
            if this_tag != tag:
                ptrace.append(this_tag)
                tag = this_tag
            content = pt.group(2)

    def _getSubUnits(self, pass_to=self.content, seen_it=[]):
        '''This differs from getting
        SubUnits from index units in that
        we have bullet points that a) have no header and b_
        have a natural language context
        to be analyzed'''
        point_heads = list(SUBITEM.finditer(self.raw))

        last_point = None
        while point_heads:
                # collect point data
            head = point_heads.pop(0)
            bpoint = head.group(1)
            clause = head.group(2)
            point_type = self._getBpointType(bpoint)

            # lookahead to see if
            # we have any more on this level.
            if point_heads:
                next_point = point_heads[0].group(1)
                next_type = self._getBpointType(next_point)
            else:
                break

            # the next 
            if point_type == last_type:
                point_at[bpoint] = SemanticUnit(clause)
            # if we've seen this type before,
            # but it isn't the current type
            # the outer scope execution of this
            # function covers the next bullet,
            # so break
            elif next_type in seen_it:
                break
            # we have a subunit
            else:
                seen_it.append(next_type)
                self._getSubUnits(pass_to=point_at[bpoint],
                                  seen_it=seen_it,
                                  last_type=point_type)

    def _mergeHeadItemSpans(self):
        out = []
        subheads = list(SUBITEM_HEAD.finditer(self.raw))
        subitems = list(POINT.finditer(self.raw))
        sbh = (True, next(subheads))
        sbi = (False, next(subitems))
        while subheads or subitems:
            if sbh <= sbi:
                out.append(sbh)
                sbh = (True, next(subheads))
            else:
                out.append(sbi)
                sbi = (False, next(subitems))
        return out

    def ParseFormatting(self):

        form = self._mergeHeadItemSpans()
        depth = 0
        bpoint = "TOP"
        items = list()
        struct_ptr = items
        for i in range(len(form)):
            is_header, match = form[i]
            if is_header:
                depth += 1
                struct_ptr = list()
                items.append(struct_ptr)
                if bpoint is "TOP":
                    # assume the entire preceding text in the passage is
                    # the context
                    # TODO(do we want a formatpoint subclass of re.match?)
                          # it would improve readability
                    context_slice = slice(0, form[0][1].start)
                else:
                    # assume the last point is the context
                    context_slice = slice(form[i-1][1].start, match.start-1)
            else:
                point = self._getBpointType(match.group(1))
                if point != bpoint:
                    depth -= 1
                    struct_ptr =

        # for any paren indicated items found in text
        while subitems:
            item_head = subitems.pop(0)
            ckey = self._getBpointType(subitems.group(1))
            # number of lower level points between
            # point x, and point x+1 on one level
            peer_at = 0

            def bmatch(p): return True if self._getBpointType(
                p) == ckey else False
            if subitems:
                # if the next subitem is
                # of the same rank
                while not bmatch(subitems[peer_at].group(1)):
                    peer_at += 1
                next_match = subitems[peer_at]
                item_end = next_match.start - 1
            else:
                item_end = len(self.raw) - 1
            subheads, subitems = subitems[:peer_at], subitems[peer_at:]
            # Should we get the semantic relation here?
            rel = self._predictFRelate()
            # TODO(maybe subheads and raw text don't both need to be passed?)
            item = SemanticUnit(
                self.raw[item_head.end:item_end], subheads, rel)
            self.content.append(item)
        # If we don't wee any subitems
        if not self.content:


class SemanticUnit(TextUnit):
    '''Inherits from TextUnit, but
    adds methods for joining to other
    semantic units. Intended for 
    listed conditions, constraints, etc.'''
    pattern = SUBITEM_HEAD
    SubUnit = None

    def __init__(self, rawtext):

    def _identifySubUnitByKey(self):


class Paragraph(PassageUnit):
    pattern = NUM_PGPH

    def __init__(self, text):
        super.__init__(None, text)


class Article(IndexUnit):

    pattern = ARTICLE
    SubUnit = Paragraph

    def __init__(self, header, text):
        super().__init__(header, text, self.SubUnit)
        self.entities = set()
        self.conditions = []

    def _getEntities(self):
        raise NotImplemented


class Section(IndexUnit):

    SubUnit = Article
    pattern = SECTION

    def __init__(self, header, text):
        super().__init__(header, text, self.SubUnit)
        #print("finished processing section: {}\n".format(header))


class Chapter(IndexUnit):

    SubUnit = Section
    pattern = CHAPTER

    def __init__(self, header, text):
        super().__init__(header, text, self.SubUnit)


class FullText(IndexUnit):

    pattern = None

    def __init__(self, header, text, SubUnit):
        super().__init__(header, text, SubUnit)

    def _depth(self, dest_node=None):
        if not dest_node:
            des_node = self.leafType()
        depth = 0
        type_ptr = cls
        while type_ptr != dest_node:
            type_ptr = type_ptr.SubUnit
            depth += 1
        return depth

    def _stripInfrastructure(self):
        return " ".join(self._textsOrSubUnits())


class TppFull(FullText):

    SubUnit = Chapter

    def __init__(self, text):
        super().__init__("TPP", text, self.SubUnit)


class TppProcessor(TextProcessor):

    def __init__(self, raw):
        # keys should be letter
        # id for section, values are section objects
        self.sections = dict()
        self.process(raw)

    def _preprocess(self, raw):
        smatches = list(SECTION.finditer(raw))
        while smatches:
            # section match matches
            # the bill code and title
            section_match = smatches.pop(0)
            scode = section_match.group(1)
            header = section_match.group(2)
            # if we're not at the end,
            # assume the end of the text
            # precedes the title of the next section
            if smatches:
                text_end = smatches[0].start() - 1
            else:
                text_end = len(raw) - 1
            text = raw[section_match.end():text_end].strip()
            self.sections[scode] = Section(header, text)

    def process(self, raw):
        super().process(raw)
        # do nlp stuff here.
        # in the case of tpp, we'll
        # be iterating over articles


# if __name__ == "__main__":
with open(TEST_FILE) as tppfile:
    tpp = tppfile.read()

proc = TppFull(tpp)
print("\n\n\n")
for p in proc.chapter:
    print(p)
    chap = proc.chapter[p]
    for s in chap.section:
        print("\t"+s)
        sec = chap.section[s]

# art = sec.article['2.3']
    # print(art.header, "\n", art.doc)
