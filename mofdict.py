import numpy as np
import regex
import json
from itertools import chain

from chemdataextractor.doc import Paragraph, Heading

from doc import Document
from doc.utils import cleanup_text
from mer import get_method_paragraphs
from mof import MOF, replace_mof
from error import MerError, MofError
from utils import find_abbreviation_from_document, remove_hydrogen


class MofDictionary(object):
    """
    MOF dictionary
    """
    __version__ = '2.1.0'

    def __init__(self, mof_list, **kwargs):
        """
        Create MofDictionary from mof_list and paper information
        :param mof_list: (list) list of class <MOF>
        :param kwargs: information of MofDictionary
        - raw_elements : (list of class <chemdataextractor.doc.Paragraph> or <chemdataextractor.doc.Header>)
                         paragraphs of paper
        - metadata : (dict) metadata of paper
        - abbreviation : (dict) abbreviation of paper
        - raw_method_paragraphs : (list of string) method paragraph of paper. If not exists, automatically find
                               method paragraph from raw_elements.
        """
        if mof_list is None:
            self.mof_list = []
        else:
            self.mof_list = mof_list

        self.raw_elements = kwargs.get('raw_elements', [])
        self.metadata = kwargs.get('metadata', {})
        self.abbreviation = kwargs.get('abbreviation', {})

        if 'raw_method_paragraphs' in kwargs:
            self.raw_method_paragraphs = kwargs.get('raw_method_paragraphs')
        else:
            try:
                self.raw_method_paragraphs = get_method_paragraphs(self.raw_elements)
            except MerError:
                self.raw_method_paragraphs = []

        self._set_doi_to_mofs()

    @classmethod
    def from_file(cls, filepath, reader=None, fill_condition=True, standard_unit=True, find_abbreviation=False,
                  convert_precursor=False, character_embedidng=True):
        """
        :param filepath: (str, bytes, os.PathLike or pathlib.Path) Path of file
        :param reader: (reader.Reader) Reader of file.
        :param fill_condition:
        :param standard_unit:
        :param find_abbreviation: If True, find abbreviation of paper.
        """

        doc = Document(filepath, reader)
        elements = doc.elements
        metadata = doc.metadata

        raw_method_paragraphs = get_method_paragraphs(elements)

        # Generate MOF list
        mof_list = []
        for paragraph in raw_method_paragraphs:
            try:
                mof = MOF.from_paragraph(paragraph, classify_material=False, standard_unit=standard_unit,
                                         database=doc.database, convert_precursor=convert_precursor,
                                         character_embedding=character_embedidng, metadata=metadata)
                mof_list.append(mof)
            except MofError:
                pass

        # Processing replace precursors
        mof_list = replace_mof(mof_list)

        # Classify precursors and extra chemicals
        for mof in mof_list:
            mof.classify_material()

        # MOF must have metal precursors
        mof_list = [mof for mof in mof_list if mof.M_precursor]

        O_precursor_iter = chain.from_iterable([mof.get_material_list('O_precursor') for mof in mof_list])

        # Find abbreviation
        if find_abbreviation:
            chemical_abbreviation = find_abbreviation_from_document(doc, O_precursor_iter)
            for mof in mof_list:
                for linker in mof.O_precursor:
                    name = linker['name']
                    name = remove_hydrogen(name)
                    if name in chemical_abbreviation:
                        linker['full_name'] = chemical_abbreviation[name]
        else:
            chemical_abbreviation = {}

        return MofDictionary(mof_list, raw_elements=elements, metadata=metadata,
                             raw_method_paragraphs=raw_method_paragraphs, abbreviation=chemical_abbreviation)

    @classmethod
    def from_dict(cls, file):
        if isinstance(file, list):
            mof_list = []
            for mof_ in file:
                mof = MOF.from_dict(mof_)
                mof_list.append(mof)
            return MofDictionary(mof_list)
        else:
            raise TypeError(f'file must be list or dict, not {type(file)}')

    @classmethod
    def from_json(cls, path):
        with open(path, 'r', encoding='utf-8') as f:
            file = json.load(f)
        return MofDictionary.from_dict(file)

    @property
    def method_paragraphs(self):
        return [cleanup_text(text) for text in self.raw_method_paragraphs]

    @property
    def elements(self):
        element_list = []
        for raw_element in self.raw_elements:
            text = raw_element.text
            text = cleanup_text(text)
            text = regex.sub(r"\s+", " ", text)
            if isinstance(raw_element, Paragraph):
                element_list.append(Paragraph(text))
            elif isinstance(raw_element, Heading):
                element_list.append(Heading(text))
            else:
                raise TypeError()
        return element_list

    def __repr__(self):
        return f"""Title : {self.title}\nDoi : {self.doi}\nJournal : {self.journal}\nDate : {self.date}"""

    def __len__(self):
        return len(self.mof_list)

    def __iter__(self):
        for mof in self.mof_list:
            yield mof

    def __getitem__(self, item):
        return self.mof_list[item]

    def __delitem__(self, item):
        del self.mof_list[item]

    def __contains__(self, item):
        return item in self.mof_list

    def __bool__(self):
        if self.mof_list:
            return True
        else:
            return False

    def to_dict(self, extract_all=False):
        return [mof.to_dict(extract_all) for mof in self.mof_list]

    def _set_doi_to_mofs(self):
        if self.doi is not None:
            for mof in self.mof_list:
                mof.doi = self.doi

    def _remove_wrong_mof(self):
        for mof in self.mof_list:
            if mof.M_precursor and mof.O_precursor:
                pass
            else:
                self.mof_list.remove(mof)

    def _fill_condition(self):
        before_temp = None
        before_time = None
        for mof in self.mof_list:
            if mof.temperature:
                before_temp = mof.temperature
            if mof.time:
                before_time = mof.time

        if not self.temperature:
            self.temperature = before_temp
        if not self.time:
            self.time = before_time

    @property
    def title(self):
        return self.metadata.get('title')

    @property
    def doi(self):
        return self.metadata.get('doi')

    @property
    def url(self):
        if self.doi is None:
            return None
        else:
            return "https://doi.org/" + self.doi

    @property
    def journal(self):
        return self.metadata.get('journal')

    @property
    def author_list(self):
        return self.metadata.get('author_list')

    @property
    def date(self):
        return self.metadata.get('date')
