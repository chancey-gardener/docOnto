#!/usr/bin/python3

import re
import spacy
import sys
import logging
from spacy.attrs import ORTH, DEP, POS, LEMMA, HEAD
from spacy.pipeline import EntityRecognizer
from abc import ABC, abstractmethod


SPACY_MODEL = "en_core_web_sm"
# Load spacy model
print("loading model")
nlp = spacy.load(SPACY_MODEL)
ner = EntityRecognizer(nlp.vocab)
print("\ncomplete\n")

# Declare and compile regex objects for identifying subunits

NUM_PGPH_PAT = r'^(\d)\.\s'
POINT_PAT = r'\((.+)\)'
IPOINT_PAT = r'i+'
SUBITEM_PAT = r'^(\([a-z]\)|\d)(.*)$'
SUBITEM_HEAD_PAT = r':$\s+\(.+\)'
TYPE_MATCH = r"<class '__main__.(\w*)'>"


IPOINT = re.compile(IPOINT_PAT)
POINT = re.compile(POINT_PAT, re.MULTILINE)
NUM_PGPH = re.compile(NUM_PGPH_PAT, re.MULTILINE)
TYPE_MATCHER = re.compile(TYPE_MATCH)
SUBITEM = re.compile(SUBITEM_PAT, re.MULTILINE)
SUBITEM_HEAD = re.compile(SUBITEM_HEAD_PAT, re.MULTILINE)



class TextUnit:
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
            if type(raw) != str:
                raise ValueError(
                    "You're raw text is of type: {}".format(type(raw)))
            smatches = list(self.SubUnit.pattern.finditer(raw))
        else:
            smatches = []
        return smatches

    # @classmethod
    def _getTypeName(self):
        try:
            name = TYPE_MATCHER.search(str(type(self))).group(1)
            return name if name else None
        except AttributeError:
            return str(type(self))

    # TODO: replace this and write method to return
    # subclass of PassageUnit
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
    SubUnit = None

    def __init__(self, header, text):
        # It's ^^ THIS ^^ level where text gets mispassed!!
        super().__init__(header, text, self.SubUnit)

    def _getPackagedUnits(self, raw, smatches):
        subunits = dict()
        while smatches:
            # section match matches
            # the bill code and title
            section_match = smatches.pop(0)
            scode = section_match.group(1)

            try:
                header = section_match.group(2)
            except IndexError:
                # TODO: use logging
                print("WARNING: {} {} has no title".format(
                    self.SubUnit.__name__,
                    scode))
                header = None

            # if we're not at the end,
            # assume the end of the text
            # precedes the title of the next section
            if smatches:
                text_end = smatches[0].start() - 1
            else:
                text_end = len(raw) - 1
            # rename this var section_text
            text = raw[section_match.end():text_end].strip()
            print(self.SubUnit)
            subunits[scode] = self.SubUnit(header, text)
        return subunits

    def _getSubUnits(self, raw):
        if self.SubUnit is None:
            return None
        smatches = self._getSubHeaders(raw)
        while not smatches:
            if self.SubUnit.SubUnit is not None:
                self.SubUnit = self.SubUnit.SubUnit
            else:
                # this feels sketchy
                self._beTerminal(raw)
                break
            smatches = self._getSubHeaders(raw)
        subunits = self._getPackagedUnits(raw, smatches)
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

    def __init__(self, header, text):
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

    def _getSubUnits(self, pass_to=None,
                     seen_it=[], last_point=None,
                     point_heads=None):
        '''This differs from getting
        SubUnits from index units in that
        we have bullet points that a) have no header and b_
        have a natural language context
        to be analyzed'''
        if pass_to is None:
            pass_to = self.content
        if point_heads is not None:
            point_heads = list(SUBITEM.finditer(self.raw))

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
                                  last_type=point_type,
                                  point_heads=point_heads)



class SemanticUnit(TextUnit):
    '''Inherits from TextUnit, but
    adds methods for joining to other
    semantic units. Intended for 
    listed conditions, constraints, etc.'''
    pattern = SUBITEM_HEAD
    SubUnit = None

    def __init__(self, rawtext):
        self.rawtext = rawtext

    def _identifySubUnitByKey(self):
        raise NotImplemented