# -*- coding:utf-8 -*-
import regex
import numpy as np

from .storage import DataStorage
from doc.utils import identify_tag


class UnitStorage(DataStorage):
    unit_dimension = {"m": np.array([0, 1, 0, 0, 0, 0, 0]), 'min': np.array([0, 0, 1, 0, 0, 0, 0]),
                      "s": np.array([0, 0, 1, 0, 0, 0, 0]), 'mol': np.array([0, 0, 0, 1, 0, 0, 0]),
                      'g': np.array([1, 0, 0, 0, 0, 0, 0]), 'L': np.array([0, 3, 0, 0, 0, 0, 0]),
                      'days': np.array([0, 0, 1, 0, 0, 0, 0]), "%": np.array([0, 0, 0, 0, 0, 0, 0]),
                      "K": np.array([0, 0, 0, 0, 1, 0, 0]), 'Torr': np.array([1, -1, -2, 0, 0, 0, 0]),
                      "h": np.array([0, 0, 1, 0, 0, 0, 0]), "Å": np.array([0, 1, 0, 0, 0, 0, 0]),
                      "wt%": np.array([0, 0, 0, 0, 0, 0, 0]), "°C": np.array([0, 0, 0, 0, 1, 0, 0]),
                      "°": np.array([0, 0, 0, 0, 0, 0, 0]), 'hours': np.array([0, 0, 1, 0, 0, 0, 0]),
                      "bar": np.array([1, -1, -2, 0, 0, 0, 0]), 'l': np.array([0, 3, 0, 0, 0, 0, 0]),
                      'minute': np.array([0, 0, 1, 0, 0, 0, 0]), 'V': np.array([1, 2, -3, 0, 0, -1, 0]),
                      'A': np.array([0, 0, 0, 0, 0, 1, 0]), 'Hz': np.array([0, 0, -1, 0, 0, 0, 0]),
                      'eV': np.array([1, 1, -1, 0, 0, 0, 0]), "Ω": np.array([1, 2, -3, 0, 0, -2, 0]),
                      'weeks': np.array([0, 0, 1, 0, 0, 0, 0]), 'day': np.array([0, 0, 1, 0, 0, 0, 0]),
                      'minutes': np.array([0, 0, 1, 0, 0, 0, 0]), 'seconds': np.array([0, 0, 1, 0, 0, 0, 0]),
                      'week': np.array([0, 0, 1, 0, 0, 0, 0]), 'hour': np.array([0, 0, 1, 0, 0, 0, 0]),
                      'month': np.array([0, 0, 1, 0, 0, 0, 0]), 'cycles': np.array([0, 0, 1, 0, 0, 0, 0]),
                      'months': np.array([0, 0, 1, 0, 0, 0, 0]), 'year': np.array([0, 0, 1, 0, 0, 0, 0]),
                      'years': np.array([0, 0, 1, 0, 0, 0, 0]), "°F": np.array([0, 0, 0, 0, 1, 0, 0]),
                      "wt.%": np.array([0, 0, 0, 0, 0, 0, 0]), 'Pa': np.array([1, -1, 2, 0, 0, 0, 0]),
                      'J': np.array([1, 2, -2, 0, 0, 0, 0]), 'F': np.array([-1, -2, 4, 0, 0, 2, 0]),
                      'cd': np.array([0, 0, 0, 0, 0, 0, 1]), 'rad': np.array([0, 0, 0, 0, 0, 0, 0]),
                      "N": np.array([1, 1, -2, 0, 0, 0, 0]), 'd': np.array([0, 0, 1, 0, 0, 0, 0]),
                      'C': np.array([0, 0, 1, 0, 0, 1, 0]), 'W': np.array([1, 2, -3, 0, 0, 0, 0]),
                      "cal": np.array([1, 2, -2, 0, 0, 0, 0]), "ppm": np.array([0, 0, 0, 0, 0, 0, 0]),
                      "M": np.array([0, -3, 0, 1, 0, 0, 0]), "PPM": np.array([0, 0, 0, 0, 0, 0, 0]),
                      "cc": np.array([0, 3, 0, 0, 0, 0, 0]), "CC": np.array([0, 3, 0, 0, 0, 0, 0]),
                      'vol': np.array([0, 0, 0, 0, 0, 0, 0]), 'cycle': np.array([0, 0, 1, 0, 0, 0, 0]),
                      'atm': np.array([1, -1, -2, 0, 0, 0, 0]), 'S': np.array([-1, -2, 3, 0, 0, 2, 0]),
                      'at%': np.array([0, 0, 0, 0, 0, 0, 0]), 'sec': np.array([0, 0, 1, 0, 0, 0, 0]),
                      'drop': np.array([0, 0, 0, 0, 0, 0, 0]), "℉": np.array([0, 0, 0, 0, 1, 0, 0]),
                      'drops': np.array([0, 0, 0, 0, 0, 0, 0]), "℃": np.array([0, 0, 0, 0, 1, 0, 0]),
                     '시간': np.array([0, 0, 1, 0, 0, 0, 0])}
    unit = sorted(unit_dimension.keys(), reverse=True)

    unit_prx = ["m", "k", "G", "M", "c", "d", "n", "μ", 'µ', "T", "P", "p"]
    element = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl",
               "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As",
               "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In",
               "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb",
               "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl",
               "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk",
               "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh",
               "Fl", "Mc", "Lv", "Ts", "Og", "Uue"]

    unit_suffix = ["cat", "cats", "-cat", "[-]cat.", "[-]cat.", 'catalyst', "Cat", "CAT", r"\scat\s", "obs", r"\(STP\)",
                   "STP"]

    def __init__(self):
        super().__init__('unit', 'u')
        self._special_unit = {}
        self.UNIT_regex = ""
        self.is_unit_regex = ""
        self.unit_classification = {}
        self._count = 0
        self.unit_analysis = {}

    def add_special_unit(self, unit_name, unit_regex):
        """Add special unit (not regular form)

        :param unit_name: (str) name of special unit
        :param unit_regex: (real_string) pattern of regular expression

        * unit_regex must include group "NUM" (which is value of unit)
        ** Special unit would be replaced to -num('group_NUM')- -u******-
        *** Special unit would be defined as unit_name, dimension is [0,0,0,0,0,0,0]
            ex. unit_name = 'pH' -> -u001000- : {unit: 'pH1.0', dimension: [0,0,0,0,0,0,0]}
        """

        self._special_unit[unit_name] = unit_regex

    def _generate_unit_hash(self, unit_text, unit_cem, unit_analysis):
        if unit_text in self.unit_classification:
            class_dictionary = self.unit_classification[unit_text]
            hash_unit = class_dictionary['hash']
        else:
            hash_unit = str(self._count).zfill(3)
            class_dictionary = {'hash': hash_unit, 'sub_class': {frozenset(): "-u{}000-".format(hash_unit)}, 'count': 1}
            self.unit_classification[unit_text] = class_dictionary
            self.data_to_hash["-u{}000-".format(hash_unit)] = {'unit': unit_text, 'unit_analysis': unit_analysis,
                                                               'unit_cem': frozenset()}
            self._count += 1

        if unit_cem in class_dictionary['sub_class']:
            hash_code = class_dictionary['sub_class'][unit_cem]
        elif not unit_cem:  # unit_cem == None
            hash_code = f"-u{hash_unit}000-"
        else:
            hash_cem = str(class_dictionary['count']).zfill(3)
            hash_code = "-u{}{}-".format(class_dictionary['hash'], hash_cem)
            class_dictionary['sub_class'][unit_cem] = hash_code
            class_dictionary['count'] += 1
            self.hash_to_data[hash_code] = {'unit': unit_text, 'unit_analysis': unit_analysis, 'unit_cem': unit_cem}

        return hash_code

    @classmethod
    def _preprocessing(cls, string):
        string_revised = string

        # initial -> 1st -> 1
        string_revised = regex.sub(r"(?i)initial", "1st", string_revised)
        # room temperature
        string_revised = regex.sub(r"(?i)room temperatures?", ' 293 K ', string_revised)
        # room pressure
        string_revised = regex.sub(r"(?i)room pressures?", ' 1 atm ', string_revised)
        # ambient temperature
        string_revised = regex.sub(r"(?i)ambient temperatures?", ' 293 K ', string_revised)
        # ambient pressure
        string_revised = regex.sub(r"(?i)ambient pressures?", ' 1 atm ', string_revised)
        # ambient condition
        string_revised = regex.sub(r"(?i)ambient conditions?", '293 K and 1 atm ', string_revised)
        # STP
        string_revised = regex.sub(r"(?<=^|\s)STP(?=\s|$)", '273 K and 1 atm ', string_revised)
        # SATP
        string_revised = regex.sub(r"(?<=^|\s)SATP(?=\s|$)", '293 K and 1 atm ', string_revised)
        # ambient condition
        string_revised = regex.sub(r"(?i)ambient conditions?", '293 K and 1 atm ', string_revised)

        # over night
        string_revised = regex.sub(r"(?i)over\s?nights?", ' 12 h ', string_revised)
        # C/10 -> 0.1C
        string_revised = regex.sub(r"(?<=^|\s)C/(?P<num>\d+)(?=\s|$)", lambda t: "%.2f C" % (1 / int(t.group("num"))),
                                   string_revised)

        return string_revised

    @classmethod
    def search_unit(cls, string, unit_sub=None, repl=None):
        """find_unit('20 m2/g') -> ('m2/g', '-num- -unit-', {'-num-':20})"""

        string = cls._preprocessing(string)

        unit = cls.unit
        unit_prx = cls.unit_prx

        if isinstance(unit_sub, list):
            unit_sub = unit_sub + cls.unit_suffix + cls.element
        else:
            unit_sub = cls.unit_suffix + cls.element
        unit_sub = sorted(unit_sub, reverse=True)

        num_string = ['several', 'few', 'first', 'second', 'third', 'forth', 'fifth', 'ninth', 'twelfth', 'one', 'two',
                      'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen',
                      'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty', 'thirty',
                      'forty', 'fifty', 'sixty', 'seventy', 'ninety', 'hundred']
        num_string = sorted(num_string, reverse=True)
        num_string = r"|".join(num_string)

        num_regex_form = fr"([±×+-]?\s?(\d+\.\d+|\d+/\d+|\d+|{num_string})\s?(st|nd|rd|th)?\s?)+(?:\(\d+\))?"
        num_simple_regex_form = r"[+-]?(\d\.\d+|\d/[1-9]\d*|\d)"  # impossible like g-10

        unit_regex_form = r"|".join(unit)
        unit_prx_regex_form = r"|".join(unit_prx)
        unit_sub_regex_form = r"|".join(unit_sub)

        unit_sum_regex_form = fr"(?P<unit_prx>{unit_prx_regex_form})?(?P<unit>{unit_regex_form})\s?(?P<times>{num_simple_regex_form})?\s?(?P<unit_sub>{unit_sub_regex_form})?"
        unit_tot_regex_form = fr"({unit_sum_regex_form})(\s+|/|⋅|per)*"

        unit_re = regex.compile(
            fr"(?<=^|\s|\(|\[|\])(?P<num>{num_regex_form})\s?(?P<UNIT>({unit_tot_regex_form})+)(?=\s|$|[)(.,]|\[|\]|$)")

        def separate_unit(units):
            unit_name = units.group("UNIT").strip()
            unit_name = unit_name.replace(" ", "")
            num = units.group("num").strip()
            return f" {num} {unit_name} "

        if not repl:
            repl = separate_unit

        sub_string = unit_re.sub(repl, string)

        return sub_string

    def find_unit(self, string, unit_sub=None):
        """find_unit('20 m2/g') -> ('m2/g', '-num- -unit-', {'-num-':20})"""

        string = self._preprocessing(string)

        unit = self.unit
        unit_prx = self.unit_prx

        if isinstance(unit_sub, list):
            unit_sub = unit_sub + self.unit_suffix + self.element
        else:
            unit_sub = self.unit_suffix + self.element
        unit_sub = sorted(unit_sub, reverse=True)

        num_string = ['several', 'few', 'first', 'second', 'third', 'forth', 'fifth', 'ninth', 'twelfth', 'one', 'two',
                      'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen',
                      'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty', 'thirty',
                      'forty', 'fifty', 'sixty', 'seventy', 'ninety', 'hundred']
        num_string = sorted(num_string, reverse=True)
        num_string = r"|".join(num_string)

        num_regex_form = fr"([±×+-]?\s?(\d+\.\d+|\d+/\d+|\d+|{num_string})\s?(st|nd|rd|th)?\s?)+(?:\(\d+\))?"
        num_simple_regex_form = r"[+-]?(\d\.\d+|\d/[1-9]\d*|\d)"  # impossible like g-10

        unit_regex_form = r"|".join(unit)
        unit_prx_regex_form = r"|".join(unit_prx)
        unit_sub_regex_form = r"|".join(unit_sub)

        unit_sum_regex_form = fr"(?P<unit_prx>{unit_prx_regex_form})?(?P<unit>{unit_regex_form})\s?(?P<times>{num_simple_regex_form})?\s?(?P<unit_sub>{unit_sub_regex_form})?"
        unit_tot_regex_form = fr"({unit_sum_regex_form})(\s+|/|⋅|per)*"

        # Save unit_regex_form
        self.UNIT_regex = unit_sum_regex_form
        self.is_unit_regex = fr"(?P<unit_prx>{unit_prx_regex_form})?(?P<unit>{unit_regex_form})\s?(?P<times>{num_simple_regex_form})?"

        unit_re = regex.compile(
            fr"(?<=^|\s|\(|\[|\])(?P<num>{num_regex_form})\s?(?P<UNIT>({unit_tot_regex_form})+)(?=\s|[가-힣]|[)(.,]|\[|\]|$)")

        def save_unit(units):
            unit_name = units.group("UNIT").strip()
            num = units.group("num").strip()

            if unit_name in self.element and len(unit_name) > 1:  # remove for error case
                return " {} ".format(units.group())

            if unit_name in self.data_to_hash:
                unit_hash = self.data_to_hash[unit_name]
            else:
                unit_analysis, unit_text, unit_cem = self._get_unit_analysis(unit_name)
                unit_hash = self._generate_unit_hash(unit_text, unit_cem, unit_analysis)

                self.unit_analysis[unit_hash] = unit_analysis
                self.unit_analysis[unit_text] = unit_analysis

                self.data_to_hash[unit_name] = unit_hash

            num_hash = "-num({})-".format(num)

            return " {} {} ".format(num_hash, unit_hash)

        sub_string = unit_re.sub(save_unit, string)
        sub_string = self._find_special_unit(sub_string)

        return sub_string

    def find_unit_from_list(self, split, unit_sub=None):

        unit = self.unit
        unit_prx = self.unit_prx

        if isinstance(unit_sub, list):
            unit_sub = unit_sub + self.unit_suffix + self.element
        else:
            unit_sub = self.unit_suffix + self.element
        unit_sub = sorted(unit_sub, reverse=True)

        num_string = ['several', 'few', 'first', 'second', 'third', 'forth', 'fifth', 'ninth', 'twelfth', 'one', 'two',
                      'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen',
                      'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty', 'thirty',
                      'forty', 'fifty', 'sixty', 'seventy', 'ninety', 'hundred']
        num_string = sorted(num_string, reverse=True)
        num_string = r"|".join(num_string)

        num_regex_form = fr"([±×+-]?\s?(\d+\.\d+|\d+/\d+|\d+|{num_string})\s?(st|nd|rd|th)?\s?)+(?:\(\d+\))?"
        num_simple_regex_form = r"[+-]?(\d\.\d+|\d/[1-9]\d*|\d)"  # impossible like g-10

        unit_regex_form = r"|".join(unit)
        unit_prx_regex_form = r"|".join(unit_prx)
        unit_sub_regex_form = r"|".join(unit_sub)

        unit_sum_regex_form = fr"(?P<unit_prx>{unit_prx_regex_form})?(?P<unit>{unit_regex_form})\s?(?P<times>{num_simple_regex_form})?\s?(?P<unit_sub>{unit_sub_regex_form})?"
        unit_tot_regex_form = fr"({unit_sum_regex_form})(\s+|/|⋅|per)*"

        self.UNIT_regex = unit_sum_regex_form
        self.is_unit_regex = fr"(?P<unit_prx>{unit_prx_regex_form})?(?P<unit>{unit_regex_form})\s?(?P<times>{num_simple_regex_form})?"

        num_re = regex.compile(fr"^(?P<num>{num_regex_form})$")
        unit_re = regex.compile(fr"^(?P<UNIT>({unit_tot_regex_form})+)$")

        def save_unit(unit_):
            units = unit_re.match(unit_)
            unit_name = units.group("UNIT").strip()

            if unit_name in self.element and len(unit_name) > 1:  # remove for error case
                return " {} ".format(units.group())

            if unit_name in self.data_to_hash:
                unit_hash = self.data_to_hash[unit_name]
            else:
                unit_analysis, unit_text, unit_cem = self._get_unit_analysis(unit_name)
                unit_hash = self._generate_unit_hash(unit_text, unit_cem, unit_analysis)

                self.unit_analysis[unit_hash] = unit_analysis
                self.unit_analysis[unit_text] = unit_analysis

                self.data_to_hash[unit_name] = unit_hash
            return unit_hash

        new_split = []
        for i, word in enumerate(split):
            if num_re.match(word):
                new_split.append(f"-num({word})-")
            elif unit_re.match(word):
                try:
                    last_word = new_split[-1]
                    assert identify_tag(last_word) == 'number'
                    hash_ = save_unit(word)
                    new_split.append(hash_)
                except (AssertionError, IndexError):
                    new_split.append(word)
            else:
                new_split.append(word)

        return new_split

    def _find_special_unit(self, string):
        """find special units and sub to '-num()-' '-u******-'
        """
        def sub_func(group):
            num = group.group("NUM")
            unit_analysis = '[0.0/0.0/0.0/0.0/0.0/0.0/0.0]'
            unit_hash = self._generate_unit_hash(unit_name, None, unit_analysis)
            self.unit_analysis[unit_hash] = unit_analysis
            self.unit_analysis[unit_name] = unit_analysis
            return f"-num({num})- {unit_hash}"

        for unit_name, unit_regex in self._special_unit.items():
            string = regex.sub(unit_regex, sub_func, string)
        return string

    def _get_unit_analysis(self, text):
        def str_to_float(string):
            fraction = regex.match(r"(?<front>[+-]?\d+)/(?P<back>[1-9]\d*)", string)
            if fraction:
                return float(fraction.group("front")) / float(fraction.group("back"))
            else:
                return float(string)

        unit_total_regex_form = fr"({self.UNIT_regex})\s*(?P<divide>/|\u22C5|per)?\s*"
        unit_sep = regex.compile(unit_total_regex_form)

        unit_text = ""
        unit_analysis = np.zeros(7, dtype='float16')

        unit_chemical = set()
        dot = 1

        for u in unit_sep.finditer(text):
            if u.group("times"):
                num = str_to_float(u.group("times")) * dot
            else:
                num = float(dot)

            if u.group("unit"):
                unit_name = u.group("unit")
            else:
                unit_name = ""

            if u.group("unit_prx"):
                unit_prx = u.group("unit_prx")
            else:
                unit_prx = ""

            unit_analysis += self.unit_dimension.get(unit_name) * num
            unit_text += unit_prx + unit_name + str(num)

            unit_sub = u.group("unit_sub")

            if unit_sub:
                unit_chemical.add(unit_sub)

            if u.group("divide") in ['/', 'per']:
                dot = -1

        string_form = "/".join(["%.1f" % i for i in unit_analysis])
        string_form = f"[{string_form}]"
        return string_form, unit_text, frozenset(unit_chemical)

    def is_unit(self, string):
        """Check the string that is unit or not
                return True when string is unit. Else, return False

                >> input
                string : <str> Target string
                include_num : <bool> If True, Find units that include number. Else, Find all of units

                >> output : <bool>
                """
        if not regex.search(r"\d", string):
            return False

        unit_total_regex_form = fr"^(({self.is_unit_regex})(\s+|/|⋅|per)*)+$"
        if regex.match(unit_total_regex_form, string):
            element_re = r"|".join(self.element)
            if regex.match(fr"^(({element_re})\d?)+\d?[+-]?$", string):
                return False
            else:
                return True
        else:
            return False

    @classmethod
    def dimension_analysis(cls, unit_name):
        """>> input
            unit_name : <str> Regularized unit ('m2.0g-1.0', '%1.0', etc.)
            >> output
            <str> Dimension of unit_name ('[2.0/1.0/0.0/0.0/0.0/0.0/0.0]')
            """

        unit_analysis = np.zeros(7, dtype='float16')
        unit_re = r"|".join(cls.unit_dimension.keys())
        unit_prx = r"|".join(cls.unit_prx)

        for group in regex.finditer(fr"({unit_prx})?(?P<unit>{unit_re})(?P<num>-?\d\.\d)", unit_name):
            unit = group.group("unit")
            num = float(group.group("num"))
            unit_analysis += cls.unit_dimension.get(unit) * num

        string_form = "/".join(["%.1f" % i for i in unit_analysis])
        string_form = f"[{string_form}]"

        return string_form

    def append(self, item, **kwargs):
        raise AttributeError

    def pop(self, item):
        raise AttributeError
