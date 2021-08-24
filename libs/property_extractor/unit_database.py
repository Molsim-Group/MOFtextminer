import regex as re
from itertools import chain
import numpy as np

from .utils import Define, Check

Dimension = {"m": np.array([0, 1, 0, 0, 0, 0, 0]), 'min': np.array([0, 0, 1, 0, 0, 0, 0]),
             "s": np.array([0, 0, 1, 0, 0, 0, 0]),
             'g': np.array([1, 0, 0, 0, 0, 0, 0]), 'L': np.array([0, 3, 0, 0, 0, 0, 0]),
             'mol': np.array([0, 0, 0, 1, 0, 0, 0]),
             'days': np.array([0, 0, 1, 0, 0, 0, 0]), "%": np.array([0, 0, 0, 0, 0, 0, 0]),
             "K": np.array([0, 0, 0, 0, 1, 0, 0]),
             "h": np.array([0, 0, 1, 0, 0, 0, 0]), "Å": np.array([0, 1, 0, 0, 0, 0, 0]),
             "%": np.array([0, 0, 0, 0, 0, 0, 0]),
             "wt%": np.array([0, 0, 0, 0, 0, 0, 0]), "°C": np.array([0, 0, 0, 0, 1, 0, 0]),
             "°": np.array([0, 0, 0, 0, 0, 0, 0]),
             "bar": np.array([1, -1, -2, 0, 0, 0, 0]), 'l': np.array([0, 3, 0, 0, 0, 0, 0]),
             'Torr': np.array([1, -1, -2, 0, 0, 0, 0]),
             'hours': np.array([0, 0, 1, 0, 0, 0, 0]), 'hour': np.array([0, 0, 1, 0, 0, 0, 0]),
             'minute': np.array([0, 0, 1, 0, 0, 0, 0]),
             'V': np.array([1, 2, -3, 0, 0, -1, 0]), 'A': np.array([0, 0, 0, 0, 0, 1, 0]),
             'Hz': np.array([0, 0, -1, 0, 0, 0, 0]),
             'eV': np.array([1, 1, -1, 0, 0, 0, 0]), "Ω": np.array([1, 2, -3, 0, 0, -2, 0]),
             'weeks': np.array([0, 0, 1, 0, 0, 0, 0]),
             'minutes': np.array([0, 0, 1, 0, 0, 0, 0]), 'seconds': np.array([0, 0, 1, 0, 0, 0, 0]),
             'day': np.array([0, 0, 1, 0, 0, 0, 0]),
             'week': np.array([0, 0, 1, 0, 0, 0, 0]), 'hour': np.array([0, 0, 1, 0, 0, 0, 0]),
             'month': np.array([0, 0, 1, 0, 0, 0, 0]),
             'months': np.array([0, 0, 1, 0, 0, 0, 0]), 'year': np.array([0, 0, 1, 0, 0, 0, 0]),
             'cycles': np.array([0, 0, 1, 0, 0, 0, 0]),
             'years': np.array([0, 0, 1, 0, 0, 0, 0]), "°F": np.array([0, 0, 0, 0, 1, 0, 0]),
             "wt.%": np.array([0, 0, 0, 0, 0, 0, 0]),
             'J': np.array([1, 2, -2, 0, 0, 0, 0]), 'F': np.array([-1, -2, 4, 0, 0, 2, 0]),
             'Pa': np.array([1, -1, 2, 0, 0, 0, 0]),
             'cd': np.array([0, 0, 0, 0, 0, 0, 1]), 'rad': np.array([0, 0, 0, 0, 0, 0, 0]),
             "N": np.array([1, 1, -2, 0, 0, 0, 0]),
             'C': np.array([0, 0, 1, 0, 0, 1, 0]), 'W': np.array([1, 2, -3, 0, 0, 0, 0]),
             "cal": np.array([1, 2, -2, 0, 0, 0, 0]),
             "M": np.array([0, -3, 0, 1, 0, 0, 0]), "PPM": np.array([0, 0, 0, 0, 0, 0, 0]),
             "ppm": np.array([0, 0, 0, 0, 0, 0, 0]),
             "cc": np.array([0, 3, 0, 0, 0, 0, 0]), "CC": np.array([0, 3, 0, 0, 0, 0, 0]),
             'vol': np.array([0, 0, 0, 0, 0, 0, 0]),
             'atm': np.array([1, -1, -2, 0, 0, 0, 0]), 'S': np.array([-1, -2, 3, 0, 0, 2, 0]),
             'cycle': np.array([0, 0, 1, 0, 0, 0, 0]),
             'at%': np.array([0, 0, 0, 0, 0, 0, 0]), 'sec': np.array([0, 0, 1, 0, 0, 0, 0]),
             'drop': np.array([0, 0, 0, 0, 0, 0, 0]),
             'drops': np.array([0, 0, 0, 0, 0, 0, 0]), "℃": np.array([0, 0, 0, 0, 1, 0, 0]),
             "℉": np.array([0, 0, 0, 0, 1, 0, 0]),
             'd': np.array([0, 0, 1, 0, 0, 0, 0])}


# In[106]:


class Unit_database():
    def __init__(self, chemical=None, smallmolecule=None, ion=None):

        """set Unit_database
        # keys
        'chemical' : dictionary (chemical -> chemhash)
        'smallmolecule' : dictionary (smallmolecule -> smhash)
        'ion' : dictionary (ion -> ionhash)
        """

        self.ELEMENTS = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl",
                         "Ar", "K",
                         "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br",
                         "Kr",
                         "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te",
                         "I",
                         "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm",
                         "Yb",
                         "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
                         "Fr",
                         "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
                         "Rf",
                         "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og", "Uue"]

        self.chemical_dict = Check(chemical, {}, lambda t: isinstance(t, dict))
        self.small_molecular_dict = Check(smallmolecule, {}, lambda t: isinstance(t, dict))
        self.ion_dict = Check(ion, {}, lambda t: isinstance(t, dict))

        self.UNIT = ["days", "weeks", "hours", "minutes", "seconds", 'sec', "Ω", "day", "week", "hour", "minute",
                     "months", "month", "year",
                     "cycles", "cycle", "years", 'bar', "Torr", 'vol', "mol", "℃", "℉", 'min', "Pa", 'cd', 'rad', "eV",
                     "cal", 'PPM', 'ppm', "cc",
                     "CC", 'drop', 'drops', "m", 's', 'h', "K", "°C", "°F", "°", "g", "Hz", 'N', "V", "L", "l", "W",
                     "%", "wt%",
                     "A", "F", "Å", "J", 'C', "M", 'S', 'atm', 'at%', 'd']

        self.UNIT_PRX = ["m", "k", "G", "M", "c", "d", "n", "μ", 'µ', "T", "P", "p"]

        # self.UNIT_mid = ["\(STP\)", "STP", "cat", '/', 'per']
        self.UNIT_sub = self.ELEMENTS + ["cat", "cats", "-cat", "[-]cat.", "[-]cat.", 'catalyst', "Cat", "CAT",
                                         r"\scat\s", "obs", "\(STP\)", "STP"]
        self.UNIT_regex = ""
        self.unit_to_hash = {}
        self.unit_classification = {}
        self.hash_to_unit = {}
        self.count = 0
        self.unit_analysis = {}

        self._special_unit = {}

    def set_chemical_database(self, **kwargs):
        """set chemical database of Unit_database
        # keys
        'chemical' : dictionary (chemical -> chemhash)
        'smallmolecule' : dictionary (smallmolecule -> smhash)
        'ion' : dictionary (ion -> ionhash)
        """
        for key, value in kwargs.items():
            if key == 'chemical':
                self.chemical_dict = Check(value, {}, lambda t: isinstance(t, dict))
            elif key == 'smallmolecule':
                self.small_molecular_dict = Check(value, {}, lambda t: isinstance(t, dict))
            elif key == 'ion':
                self.ion_dict = Check(value, {}, lambda t: isinstance(t, dict))

    def add_special_unit(self, unit_name, unit_regex):
        """Add special unit (not regular form)
        >>>input
        unit_name = (str) name of special unit
        unit_regex = (real_string) pattern of regular expression
        
        * unit_regex must include group "NUM" (which is value of unit)
        ** Special unit would be replaced to -num('group_NUM')- -u******-
        *** Special unit would be defined as unit_name, dimension is [0,0,0,0,0,0,0]
            ex. unit_name = 'pH' -> -u001000- : {unit: 'pH1.0', dimension: [0,0,0,0,0,0,0]}
        """
        self._special_unit[unit_name] = unit_regex

    def __getitem__(self, name):
        if isinstance(name, str):
            if re.match(r"-u\d\d\d\d\d\d-", name):
                return self.hash_to_unit.get(name)
            else:
                return self.unit_to_hash.get(name, None)
        elif isinstance(name, tuple) or isinstance(name, list):
            return self.find_hash(name)
        else:
            raise TypeError()

    def __len__(self):
        return len(self.hash_to_unit)

    def __repr__(self):
        return str(self.unit_to_hash.keys())

    def __bool__(self):
        return True if len(self) > 0 else False

    def to_json(self):
        pass

    def from_json(self):
        pass

    def change_to_re(self):
        new_list = self.UNIT_sub
        for chemical in chain(self.chemical_dict.keys(), self.small_molecular_dict.keys(), self.ion_dict.keys()):

            if chemical in new_list:
                continue

            chemical_revised = re.sub(pattern=r"(?:\[|\]|\(|\)|\.|\,|\-|\*|\?|\{|\}|\$|\^|[|]|\+|\\)",
                                      string=chemical, repl=lambda t: "\{}".format(t.group()))

            if re.findall(pattern=chemical_revised, string=chemical):
                new_list.append(chemical_revised)
            else:
                print("re compile error", chemical)

        # should be revised
        new_list.sort(reverse=True)
        return new_list

    def unit_hash(self, unit_text, unit_cem, unit_analysis):
        if unit_text in self.unit_classification:
            class_dictionary = self.unit_classification[unit_text]
            hash_unit = class_dictionary['hash']
        else:
            hash_unit = str(self.count).zfill(3)
            class_dictionary = {'hash': hash_unit, 'sub_class': {frozenset(): "-u{}000-".format(hash_unit)}, 'count': 1}
            self.unit_classification[unit_text] = class_dictionary
            self.hash_to_unit["-u{}000-".format(hash_unit)] = {'unit': unit_text, 'unit_analysis': unit_analysis,
                                                               'unit_cem': frozenset()}
            self.count += 1

        if unit_cem in class_dictionary['sub_class']:
            hash_code = class_dictionary['sub_class'][unit_cem]
        elif not unit_cem:  # unit_cem == None
            hash_code = f"-u{hash_unit}000-"
        else:
            hash_cem = str(class_dictionary['count']).zfill(3)
            hash_code = "-u{}{}-".format(class_dictionary['hash'], hash_cem)
            class_dictionary['sub_class'][unit_cem] = hash_code
            class_dictionary['count'] += 1
            self.hash_to_unit[hash_code] = {'unit': unit_text, 'unit_analysis': unit_analysis, 'unit_cem': unit_cem}

        return hash_code

    def find_hash(self, *args):
        lens = len(args)
        if lens == 1:
            if (isinstance(args[0], tuple) or isinstance(args[0], list)) and len(args[0]) == 2:
                unit_text = args[0][0]
                unit_cem = args[0][1]
        elif lens == 2:
            unit_text = args[0]
            unit_cem = args[1]
        else:
            raise TypeError
        try:
            return self.unit_classification[unit_text]['sub_class'][unit_cem]
        except KeyError:
            return None

    def is_unit(self, string, include_num=False):
        """Check the string that is unit or not
        return True when string is unit. Else, return False
        
        >> input
        string : <str> Target string
        include_num : <bool> If True, Find units that include number. Else, Find all of units
        
        >> output : <bool>
        """
        if not re.search(r"\d", string):
            return False

        UNIT_total_regex_form = fr"^(({self.is_unit_regex})(\s+|/|⋅|per)*)+$"
        if re.match(UNIT_total_regex_form, string):
            ELEMENT_re = r"|".join(self.ELEMENTS)
            # print (string, re.match(fr"^(({ELEMENT_re})\d?)+\d?[+-]?$", string))
            if re.match(fr"^(({ELEMENT_re})\d?)+\d?[+-]?$", string):
                return False
            else:
                return True
        else:
            return False

    def _func_hash(self, num_string, unit):
        if unit in self.unit_to_hash:
            unit_hash = self.unit_to_hash[unit]
        else:
            unit_analysis = dimension_analysis(unit)
            unit_hash = self.unit_hash(unit, None, unit_analysis)

            self.unit_analysis[unit_hash] = unit_analysis
            self.unit_analysis[unit] = unit_analysis

        return f" -num({num_string})- {unit_hash} "

    def preprocessing(self, string):
        string_revised = string
        # initial -> 1st -> 1
        string_revised = re.sub(r"(?i)initial", "1st", string_revised)
        # room temperature
        string_revised = re.sub(r"(?i)room temperatures?", self._func_hash('room temperature', 'K1.0'), string_revised)
        # room pressure
        string_revised = re.sub(r"(?i)room pressures?", self._func_hash('room pressure', 'atm1.0'), string_revised)

        return string_revised

    def find_unit(self, string, UNIT_sub=None):
        """find_unit('20 m2/g') -> ('m2/g', '-num- -unit-', {'-num-':20})"""

        string = self.preprocessing(string)

        UNIT = self.UNIT
        UNIT_PRX = self.UNIT_PRX
        # UNIT_mid = self.UNIT_mid
        if not UNIT_sub:
            UNIT_sub = self.change_to_re()

        # num_string = r"(?:[±×+-]?\s?\d+[.]?\d*(?:[(]\d+[)])?\s?)+"
        num_string = "several few first second third forth fifth ninth twelfth one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen ninteen twenty thirty fourty fifty sixty seventy ninety hundred".split()
        num_string = r"|".join(num_string)

        # num_regex_form = fr"(?:[±×+-]?\s?\d+[.]?\d*(?:[(]\d+[)])?\s?|(?:{num_string})\s?)+(?:st|nd|rd|th)?\s?)"
        num_regex_form = fr"([±×+-]?\s?(\d+\.\d+|\d+/\d+|\d+|{num_string})\s?(st|nd|rd|th)?\s?)+(?:\(\d+\))?"
        num_simple_regex_form = r"[+-]?(\d\.\d+|\d/[1-9]\d*|\d)"  # impossible like g-10

        unit_regex_form = r"|".join(UNIT)
        unit_prx_regex_form = r"|".join(UNIT_PRX)
        # unit_mid_regex_form = r"|".join(UNIT_mid)
        unit_sub_regex_form = r"|".join(UNIT_sub)

        # UNIT_sum_regex_form = fr"(?P<unit_prx>{unit_prx_regex_form})?(?P<unit>{unit_regex_form})\s?(?P<unit_sub>{unit_sub_regex_form})?(?P<times>{num_simple_regex_form})?\s?({unit_mid_regex_form})?"
        UNIT_sum_regex_form = fr"(?P<unit_prx>{unit_prx_regex_form})?(?P<unit>{unit_regex_form})\s?(?P<times>{num_simple_regex_form})?\s?(?P<unit_sub>{unit_sub_regex_form})?"
        UNIT_total_regex_form = fr"({UNIT_sum_regex_form})(\s+|/|⋅|per)*"

        self.UNIT_regex = UNIT_sum_regex_form
        self.is_unit_regex = fr"(?P<unit_prx>{unit_prx_regex_form})?(?P<unit>{unit_regex_form})\s?(?P<times>{num_simple_regex_form})?"

        # UNIT_re = re.compile(r"(?<=\s|^|[({]|\[|\])(?P<num>(?:[±×+-]?\s?\d+[.]?\d*(?:[(]\d+[)])?\s?|(?:"+num_string+")\s?)+(?:st|nd|rd|th)?\s?)(?P<UNIT>(?:(?:" + "|".join(UNIT_PRX)+r")?(?:" + r"|".join(UNIT)+r")\s?(?:(?:" + r"|".join(UNIT_sub)+r")\d?[+-]?)*(?:[-]?\d*)\s?(?:" + "|".join(UNIT_mid)+r")*(\?:\s{0,2}[⋅]\s{0,2}|\s)?)+)(?=\s|$|[)(}{.,]|\[|\]|$)")
        UNIT_re = re.compile(
            fr"(?<=^|\s|\(|\[|\])(?P<num>{num_regex_form})\s?(?P<UNIT>({UNIT_total_regex_form})+)(?=\s|$|[)(.,]|\[|\]|$)")

        sub_string = string

        UNIT_list = set()

        def regex_unit_func(units):

            unit = units.group("UNIT").strip()
            num = units.group("num").strip()

            if unit in self.UNIT_sub and len(unit) > 1:
                return " {} ".format(units.group())

            if unit in self.unit_to_hash:
                unit_hash = self.unit_to_hash.get(unit)

            else:
                unit_analysis, unit_text, unit_cem = self._unit_analysis_func(unit, UNIT_sub)
                unit_hash = self.unit_hash(unit_text, unit_cem, unit_analysis)

                self.unit_analysis[unit_hash] = unit_analysis
                self.unit_analysis[unit_text] = unit_analysis

                self.unit_to_hash[unit] = unit_hash

            UNIT_list.add(unit_hash)

            num_hash = "-num({})-".format(num)

            return " {} {} ".format(num_hash, unit_hash)

        sub_string = UNIT_re.sub(regex_unit_func, string)

        sub_string = self._find_special_unit(sub_string)

        return UNIT_list, sub_string

    def _find_special_unit(self, sub_string):
        """find special units and sub to '-num()-' '-u******-'
        """

        def sub_func(group):
            num = group.group("NUM")
            unit_analysis = '[0.0/0.0/0.0/0.0/0.0/0.0/0.0]'
            unit_hash = self.unit_hash(unit_name, None, unit_analysis)

            self.unit_analysis[unit_hash] = unit_analysis
            self.unit_analysis[unit_name] = unit_analysis

            return f"-num({num})- {unit_hash}"

        for unit_name, unit_regex in self._special_unit.items():
            sub_string = re.sub(unit_regex, sub_func, sub_string)

        return sub_string

    def _unit_analysis_func(self, text, UNIT_sub):  # Mass, Length, Time, Num, Temp, Amphere, Candela
        global Dimension
        """Dimension = {"m" : np.array([0,1,0,0,0,0,0]),'min':np.array([0,0,1,0,0,0,0]), "s":np.array([0,0,1,0,0,0,0]), 
             'g':np.array([1,0,0,0,0,0,0]), 'L':np.array([0,3,0,0,0,0,0]), 'mol':np.array([0,0,0,1,0,0,0]),
             'days':np.array([0,0,1,0,0,0,0]), "%":np.array([0,0,0,0,0,0,0]), "K":np.array([0,0,0,0,1,0,0]), 
             "h":np.array([0,0,1,0,0,0,0]), "Å": np.array([0,1,0,0,0,0,0]), "%":np.array([0,0,0,0,0,0,0]),
             "wt%":np.array([0,0,0,0,0,0,0]), "°C":np.array([0,0,0,0,1,0,0]), "°":np.array([0,0,0,0,0,0,0]),
             "bar":np.array([1,-1,-2,0,0,0,0]), 'l':np.array([0,3,0,0,0,0,0]), 'Torr':np.array([1,-1,-2,0,0,0,0]), 
             'hours':np.array([0,0,1,0,0,0,0]), 'hour':np.array([0,0,1,0,0,0,0]), 'minute':np.array([0,0,1,0,0,0,0]),
            'V' : np.array([1,2,-3,0,0,-1,0]), 'A':np.array([0,0,0,0,0,1,0]), 'Hz':np.array([0,0,-1,0,0,0,0]),
             'eV':np.array([1,1,-1,0,0,0,0]), "Ω":np.array([1,2,-3,0,0,-2,0]), 'weeks':np.array([0,0,1,0,0,0,0]),
            'minutes':np.array([0,0,1,0,0,0,0]), 'seconds':np.array([0,0,1,0,0,0,0]),'day':np.array([0,0,1,0,0,0,0]),
            'week':np.array([0,0,1,0,0,0,0]),'hour':np.array([0,0,1,0,0,0,0]),'month':np.array([0,0,1,0,0,0,0]),
            'months':np.array([0,0,1,0,0,0,0]), 'year':np.array([0,0,1,0,0,0,0]), 'cycles':np.array([0,0,1,0,0,0,0]),
            'years':np.array([0,0,1,0,0,0,0]), "°F":np.array([0,0,0,0,1,0,0]), "wt.%":np.array([0,0,0,0,0,0,0]),
            'J':np.array([1,2,-2,0,0,0,0]), 'F':np.array([-1,-2,4,0,0,2,0]), 'Pa':np.array([1,-1,2,0,0,0,0]), 
            'cd':np.array([0,0,0,0,0,0,1]), 'rad':np.array([0,0,0,0,0,0,0]), "N":np.array([1,1,-2,0,0,0,0]),
            'C':np.array([0,0,1,0,0,1,0]), 'W':np.array([1,2,-3,0,0,0,0]), "cal":np.array([1,2,-2,0,0,0,0]),
            "M":np.array([0,-3,0,1,0,0,0]), "PPM":np.array([0,0,0,0,0,0,0]), "ppm":np.array([0,0,0,0,0,0,0]),
            "cc":np.array([0,3,0,0,0,0,0]), "CC":np.array([0,3,0,0,0,0,0]), 'vol':np.array([0,0,0,0,0,0,0]),
            'atm':np.array([1,-1,-2,0,0,0,0]), 'S':np.array([-1,-2,3,0,0,2,0]), 'cycle':np.array([0,0,1,0,0,0,0]),
            'at%':np.array([0,0,0,0,0,0,0]), 'sec':np.array([0,0,1,0,0,0,0]), 'drop':np.array([0,0,0,0,0,0,0]),
                    'drops':np.array([0,0,0,0,0,0,0])}"""

        UNIT_total_regex_form = fr"({self.UNIT_regex})\s*(?P<divide>/|⋅|per)?\s*"
        UNIT_SEP = re.compile(UNIT_total_regex_form)

        unit_text = ""
        unit_analysis = np.zeros(7, dtype='float16')

        unit_chemical = set()
        dot = 1

        for u in UNIT_SEP.finditer(text):

            num = Define(u.group("times"), float(dot), lambda t: self._str_to_float(t) * dot)
            unit_name = Define(u.group("unit"), "")
            unit_prx = Define(u.group("unit_prx"), "")

            unit_analysis += Dimension.get(unit_name) * num
            unit_text += unit_prx + unit_name + str(num)

            unit_sub = u.group("unit_sub")

            if unit_sub:
                unit_chemical.add(unit_sub)

            if u.group("divide"):
                dot = -1
                # continue

        string_form = "/".join(["%.1f" % i for i in unit_analysis])
        string_form = f"[{string_form}]"
        return string_form, unit_text, frozenset(unit_chemical)

    def _str_to_float(self, t):
        fraction = re.match(r"(?<front>[+-]?\d+)/(?P<back>[1-9]\d*)", t)
        if fraction:
            return float(fraction.group("front")) / float(fraction.group("back"))
        else:
            return float(t)


def dimension_analysis(unit_name):
    """>> input
    unit_name : <str> Regularized unit ('m2.0g-1.0', '%1.0', etc.)
    >> output
    <str> Dimension of unit_name ('[2.0/1.0/0.0/0.0/0.0/0.0/0.0]')
    """
    global Dimension
    """Dimension = {"m" : np.array([0,1,0,0,0,0,0]),'min':np.array([0,0,1,0,0,0,0]), "s":np.array([0,0,1,0,0,0,0]), 
             'g':np.array([1,0,0,0,0,0,0]), 'L':np.array([0,3,0,0,0,0,0]), 'mol':np.array([0,0,0,1,0,0,0]),
             'days':np.array([0,0,1,0,0,0,0]), "%":np.array([0,0,0,0,0,0,0]), "K":np.array([0,0,0,0,1,0,0]), 
             "h":np.array([0,0,1,0,0,0,0]), "Å": np.array([0,1,0,0,0,0,0]), "%":np.array([0,0,0,0,0,0,0]),
             "wt%":np.array([0,0,0,0,0,0,0]), "°C":np.array([0,0,0,0,1,0,0]), "°":np.array([0,0,0,0,0,0,0]),
             "bar":np.array([1,-1,-2,0,0,0,0]), 'l':np.array([0,3,0,0,0,0,0]), 'Torr':np.array([1,-1,-2,0,0,0,0]), 
             'hours':np.array([0,0,1,0,0,0,0]), 'hour':np.array([0,0,1,0,0,0,0]), 'minute':np.array([0,0,1,0,0,0,0]),
            'V' : np.array([1,2,-3,0,0,-1,0]), 'A':np.array([0,0,0,0,0,1,0]), 'Hz':np.array([0,0,-1,0,0,0,0]),
             'eV':np.array([1,1,-1,0,0,0,0]), "Ω":np.array([1,2,-3,0,0,-2,0]), 'weeks':np.array([0,0,1,0,0,0,0]),
            'minutes':np.array([0,0,1,0,0,0,0]), 'seconds':np.array([0,0,1,0,0,0,0]),'day':np.array([0,0,1,0,0,0,0]),
            'week':np.array([0,0,1,0,0,0,0]),'hour':np.array([0,0,1,0,0,0,0]),'month':np.array([0,0,1,0,0,0,0]),
            'months':np.array([0,0,1,0,0,0,0]), 'year':np.array([0,0,1,0,0,0,0]), 'cycles':np.array([0,0,1,0,0,0,0]),
            'years':np.array([0,0,1,0,0,0,0]), "°F":np.array([0,0,0,0,1,0,0]), "wt.%":np.array([0,0,0,0,0,0,0]),
            'J':np.array([1,2,-2,0,0,0,0]), 'F':np.array([-1,-2,4,0,0,2,0]), 'Pa':np.array([1,-1,2,0,0,0,0]), 
            'cd':np.array([0,0,0,0,0,0,1]), 'rad':np.array([0,0,0,0,0,0,0]), "N":np.array([1,1,-2,0,0,0,0]),
            'C':np.array([0,0,1,0,0,1,0]), 'W':np.array([1,2,-3,0,0,0,0]), "cal":np.array([1,2,-2,0,0,0,0]),
            "M":np.array([0,-3,0,1,0,0,0]), "PPM":np.array([0,0,0,0,0,0,0]), "ppm":np.array([0,0,0,0,0,0,0]),
            "cc":np.array([0,3,0,0,0,0,0]), "CC":np.array([0,3,0,0,0,0,0]), 'vol':np.array([0,0,0,0,0,0,0]),
            'atm':np.array([1,-1,-2,0,0,0,0]), 'S':np.array([-1,-2,3,0,0,2,0]), 'cycle':np.array([0,0,1,0,0,0,0]),
            'at%':np.array([0,0,0,0,0,0,0]), 'sec':np.array([0,0,1,0,0,0,0]), 'drop':np.array([0,0,0,0,0,0,0]),
                    'drops':np.array([0,0,0,0,0,0,0])}"""

    unit_analysis = np.zeros(7, dtype='float16')
    unit_re = r"|".join(Dimension.keys())
    unit_prx = r"|".join(["m", "k", "G", "M", "c", "d", "n", "μ", 'µ', "T", "P", "p"])

    for group in re.finditer(fr"({unit_prx})?(?P<unit>{unit_re})(?P<num>-?\d\.\d)", unit_name):
        unit = group.group("unit")
        num = float(group.group("num"))
        # print (unit, num)

        unit_analysis += Dimension.get(unit) * num

    string_form = "/".join(["%.1f" % i for i in unit_analysis])
    string_form = f"[{string_form}]"

    return string_form
