from functools import reduce
import numpy as np
import regex
from pathlib import Path
import json

from chemdataextractor.doc import Paragraph
from fuzzywuzzy import fuzz

from doc.utils import _change_to_regex_string
from error import DatabaseError


class CounterABB(object):
    def __init__(self, abb_type=None, trainable=True):
        self.count = np.zeros(2, dtype='int16')
        self.Trainable = True
        self.update(abb_type, trainable)
        self.ABB_type = None

    def __repr__(self):
        return "({}/{})".format(self.count[0], self.count[1])

    def update(self, abb_type=None, trainable=True):
        if not trainable:
            self.Trainable = False
            self.ABB_type = abb_type
            if abb_type:
                self.count = np.array([0, -1], dtype='int8')
            else:
                self.count = np.array([-1, 0], dtype='int8')

        if not self.Trainable:
            return self.ABB_type

        if isinstance(abb_type, list):
            for types in abb_type:
                self.update(types)

        if self.checking(abb_type):
            self.count[0] += 1
        else:
            self.count[1] += 1

        self.ABB_type = self.abb_type_checking()
        return self.ABB_type

    def abb_type_checking(self):
        label = ["CM", None]
        index = int(np.argmax(self.count))
        return label[index]

    @staticmethod
    def checking(types=None):
        if types == "CM":
            return True
        else:
            return False


class Abbreviation(object):
    def __init__(self, abb_name, abb_def, abb_type_original=None, trainable=True):
        self.ABB_def = abb_def
        self.ABB_name = abb_name

        self.ABB_type = None
        self.ABB_class, self.ABB_class_type = [], []
        self.update(abb_def, abb_type_original, trainable=trainable)

    @staticmethod
    def _check_abb_type(abb_def):

        checking_string = abb_def
        checking_string = regex.sub(r"\b-\b", " - ", checking_string)
        checking = Paragraph(checking_string).cems

        if checking:
            abb_type = 'CM'
        else:
            abb_type = None
        return abb_type

    def _check_validation(self, abb_def, abb_type):
        abb_name = self.ABB_name
        abb_front_char = reduce(lambda x, y: x + "".join(regex.findall(r"^\S|[A-Z]", y)),
                                regex.split(r",|\s|-", abb_def), "")
        if abb_name[-1] == 's' and abb_def[-1] == 's':
            abb_front_char += 's'
        ratio = fuzz.ratio(abb_name.lower(), abb_front_char.lower())

        return ratio > 70 or abb_type == 'CM'

    def __repr__(self):
        return "(" + ") / (".join(self.ABB_class) + ")"

    def __eq__(self, other):
        if isinstance(other, str):
            return self.ABB_name == other
        elif isinstance(other, Abbreviation):
            return self.ABB_name == other.ABB_name
        else:
            return False

    def __ne__(self, other):
        return not self == other

    def __len__(self):
        return len(self.ABB_class)

    def __getitem__(self, key):
        return self.ABB_class[key]

    def get(self, key, default=None):
        try:
            return self.__getitem__(key)
        except KeyError:
            return default

    def change_abb_type(self, abb_def, abb_type):
        self.update(abb_def, abb_type, trainable=False)

    def update(self, abb_def, abb_type_original=None, trainable=True):
        abb_type = abb_type_original

        if isinstance(abb_def, list):
            for def_ in abb_def:
                self.update(def_, abb_type)
            return

        for i, classification in enumerate(self.ABB_class):
            result = self.compare(abb_def, classification)
            if result > 70:
                self.ABB_class_type[i].update(abb_type, trainable=trainable)

                self.ABB_def = classification
                self.ABB_type = self.ABB_class_type[i].ABB_type

                return None

        self.ABB_class.append(abb_def)

        self.ABB_class_type.append(CounterABB(abb_type, trainable))

        self.ABB_def = abb_def
        self.ABB_type = abb_type

    @staticmethod
    def compare(text1, text2):
        return fuzz.ratio(text1.lower(), text2.lower())


class AbbStorage(object):
    def __init__(self):
        self.name_to_abb = {}
        self.def_to_name = {}
        self.name = "abbreviation"

    def __repr__(self):
        return self.name

    def __len__(self):
        return len(self.name_to_abb)

    def __getitem__(self, item):
        if item in self.name_to_abb:
            return self.name_to_abb[item]
        elif item in self.def_to_name:
            return self.def_to_name[item]
        else:
            raise DatabaseError(f'{item} not in {self.name} storage')

    def __contains__(self, item):
        return item in self.def_to_name or item in self.name_to_abb

    def get(self, k, d=None):
        try:
            return self[k]
        except KeyError:
            return d

    def get_abbreviation(self, item, d=None):
        try:
            abb_name = self.def_to_name[item]
            return self.name_to_abb[abb_name]
        except IndexError:
            return d

    def get_name(self, item, d=None):
        try:
            return self.def_to_name[item]
        except IndexError:
            return d

    def keys(self):
        return self.name_to_abb.keys()

    def values(self):
        return self.name_to_abb.values()

    def items(self):
        return self.name_to_abb.items()

    def append(self, abb_name, abb_def, abb_type, trainable=True):
        if abb_name in self.name_to_abb:
            new_abb = self.name_to_abb[abb_name]
            new_abb.update(abb_def, abb_type, trainable)
            self.def_to_name[abb_def] = abb_name

        else:
            new_abb = Abbreviation(abb_name, abb_def, abb_type, trainable)
            if not len(new_abb):
                return None
            self.name_to_abb[abb_name] = new_abb
            self.def_to_name[abb_name] = abb_name
            self.def_to_name[abb_def] = abb_name

        return new_abb

    @property
    def abb_regex(self):
        regex_pattern = _change_to_regex_string(self.def_to_name.keys(), return_as_str=True)
        return regex_pattern


def read_abbreviation_from_json(path, trainable=False):
    """
    abb_database = read_abbreviation_from_json(file_path)

    json file must be list of tuple -> [(ABB_name, ABB_definition, ABB_type), .. ]

    :param path: path of json
    :param trainable: If True, type of abbreviation can be changed. (False is recommended)
    :return: <class MOFDICT.doc.storage.AbbStorage>
    """
    path_ = Path(str(path))
    if not path_.exists():
        raise FileNotFoundError
    elif path_.suffix not in ['.json']:
        raise TypeError(f'expected json, but {path_.suffix}')

    with open(str(path_), 'r') as f:
        list_of_abbreviation = json.load(f)

    return read_abbreviation_from_list(list_of_abbreviation, trainable)


def read_abbreviation_from_list(list_of_abbreviation, trainable=False):
    """
    abb_database = read_abbreviation_from_file([('ASAP', 'as soon as possible', None),
                                                ('DKL', 'depolymerised Kraft lignin', 'CM')])

    :param list_of_abbreviation: (json) list of tuple (ABB_name, ABB_definition, ABB_type).
                                        ABB_type must be None or 'CM'
    :param trainable: If True, type of abbreviation can be changed. (False is recommended)
    :return: <class MOFDICT.doc.storage.AbbStorage>
    """

    storage = AbbStorage()
    for abb_tuple in list_of_abbreviation:
        if len(abb_tuple) != 3:
            raise TypeError('input must be list of tuple : (ABB_name, ABB_definition, ABB_type)')
        abb_name, abb_def, abb_type = abb_tuple
        if isinstance(abb_name, str) and isinstance(abb_def, str):
            storage.append(abb_name, abb_def, abb_type, trainable)
        else:
            raise TypeError('input must be list of tuple : (ABB_name, ABB_definition, ABB_type)')

    return storage
