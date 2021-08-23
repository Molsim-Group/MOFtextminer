import regex
from collections import defaultdict
from numpy import ndarray, zeros

from doc import Document
from doc.utils import cleanup_text, _change_to_regex_string


class ComputableNamedList(ndarray):
    def __new__(cls, variable, default=0):
        if isinstance(variable, (list, tuple)):
            var = variable
        elif isinstance(variable, str):
            var = variable.split()
        else:
            raise TypeError()

        len_ = len(var)

        obj = zeros(len_).view(cls)
        obj.fill(default)
        obj.variable = var

        return obj

    def __getattr__(self, obj):
        try:
            index = self.variable.index(obj)
            return self[index]

        except ValueError:
            raise AttributeError()

    def __setattr__(self, obj, value):
        if obj == 'variable':
            super(ComputableNamedList, self).__setattr__(obj, value)
        else:
            try:
                index = self.variable.index(obj)
                self[index] = value

            except ValueError:
                raise AttributeError()


def transform_unit(value_and_unit, float_type='str', return_type='tuple'):
    def get_float(num):
        number = {'one': 1, 'first': 1, 'two': 2, 'second': 2, 'three': 3, 'third': 3, 'four': 4, 'forth': 4, 'five': 5,
                  'fifth': 5, 'six': 6,
                  'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13,
                  'fourteen': 14,
                  'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20}

        if num.lower() in number:
            new_value = number[num]
        else:
            regex_float = regex.match(r"^-?[0-9.]+", num)

            if regex_float:
                new_value = float(regex_float.group())
            else:
                new_value = None

        return new_value

    transform_function = {'°C1.0': [lambda t: get_float(t) + 273.15, 'K1.0'],
                          '℃1.0': [lambda t: get_float(t) + 273.15, 'K1.0'],
                          'min1.0': [lambda t: get_float(t) / 60, 'h1.0'],
                          'minutes1.0': [lambda t: get_float(t) / 60, 'h1.0'],
                          'sec1.0': [lambda t: get_float(t) / 3600, 'h1.0'],
                          'second1.0': [lambda t: get_float(t) / 3600, 'h1.0'],
                          's1.0': [lambda t: get_float(t) / 3600, 'h1.0'],
                          'hour1.0': [lambda t: get_float(t), 'h1.0'],
                          'hours1.0': [lambda t: get_float(t), 'h1.0'],
                          'day1.0': [lambda t: 24 * get_float(t), 'h1.0'],
                          'days1.0': [lambda t: 24 * get_float(t), 'h1.0'],
                          'd1.0': [lambda t: 24 * get_float(t), 'h1.0'],
                          'week1.0': [lambda t: 7 * 24 * get_float(t), 'h1.0'],
                          'weeks1.0': [lambda t: 7 * 24 * get_float(t), 'h1.0'],
                          'month1.0': [lambda t: 30 * 7 * 24 * get_float(t), 'h1.0'],
                          'months1.0': [lambda t: 30 * 7 * 24 * get_float(t), 'h1.0'],
                          'mmol1.0': [lambda t: 0.001 * get_float(t), 'mol1.0'],
                          'mg1.0': [lambda t: 0.001 * get_float(t), 'g1.0'],
                          'mL1.0': [lambda t: 0.001 * get_float(t), 'L1.0'],
                          'cm3.0': [lambda t: 0.001 * get_float(t), 'L1.0'],
                          'cc1.0': [lambda t: 0.001 * get_float(t), 'L1.0']}

    if not value_and_unit:
        return None, None
    if float_type not in ['float', 'str']:
        raise TypeError()

    elif isinstance(value_and_unit, (dict, defaultdict)):
        value = value_and_unit['Value']
        unit = value_and_unit['Unit']
    elif isinstance(value_and_unit, (tuple, list)):
        value = value_and_unit[0]
        unit = value_and_unit[1]
    else:
        raise TypeError

    new_unit = unit
    transform_value = value

    if unit in transform_function:
        function, new_unit = transform_function.get(unit)
        try:
            transform_value = function(value)
        except (KeyError, ValueError, TypeError):
            new_unit = unit

    if float_type == 'str':
        transform_value = str(transform_value)
    elif float_type == 'float':
        try:
            transform_value = float(transform_value)
        except (KeyError, ValueError, TypeError):
            transform_value = None

    if return_type == 'tuple':
        return transform_value, new_unit
    elif return_type == 'dict':
        return {'Value': transform_value, 'Unit': new_unit}
    else:
        raise ValueError(f'return_type must be tuple or dict, not {return_type}')


def find_abbreviation_from_document(doc: Document, precursor_list=None):
    chem_dict = {}
    chemical_regex = regex.compile(r'-(c|s|i)\d{6,7}-|-strange(.+)-|\S{0,2}L\S{0,2}$')
    precursor_set = {remove_hydrogen(material) for material in precursor_list}
    precursor_string = _change_to_regex_string(precursor_set, return_as_str=True)
    # print (precursor_string)
    L_search1 = fr"(?P<name>\)|\(|=|,)(^|\s+)(?:H\d?)?(?:{precursor_string})(?:H\d?)?(\s|$)"
    L_search2 = fr"(?<=^|\s)(?:H\d?)?(?:{precursor_string})(?:H\d?)?(\s|$)(?P<name>\)|\(|=|,)"

    L_regex1 = regex.compile(L_search1)
    L_regex2 = regex.compile(L_search2)

    for element in doc.elements:
        text = cleanup_text(element.text)
        if not L_regex1.search(text) and not L_regex2.search(text):
            continue

        list_of_word = doc.tokenize(element)
        if not list_of_word:
            continue

        for i, word in enumerate(list_of_word):
            if word not in ["=", "(", '[', '{']:
                continue
            try:
                before_word = list_of_word[i - 1]
                after_word = list_of_word[i + 1]
            except IndexError:
                continue

            if chemical_regex.match(before_word) and chemical_regex.match(after_word):  # Change
                before_ = doc.get_name(before_word)
                after_ = doc.get_name(after_word)

                if before_ == after_:
                    if before_ in doc.database['abbreviation']:
                        fullname = doc.database['abbreviation'].get_abbreviation(before_).ABB_def
                        if before_ != fullname:
                            before_ = remove_hydrogen(before_)
                            chem_dict[before_] = fullname

                elif len(before_) > len(after_):
                    after_ = remove_hydrogen(after_)
                    if after_ not in chem_dict:
                        chem_dict[after_] = before_
                else:
                    before_ = remove_hydrogen(before_)
                    if before_ not in chem_dict:
                        chem_dict[before_] = after_
    return chem_dict


def remove_hydrogen(linker):
    return regex.sub(r"^H\d?|H\d?$", "", linker)
