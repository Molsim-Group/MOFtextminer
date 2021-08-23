from pathlib import Path
import regex
from collections import Counter
from itertools import chain, tee
from functools import reduce

from chemdataextractor.doc import Paragraph
from fuzzywuzzy import fuzz

from reader import Default_readers
from doc.storage import DataStorage, UnitStorage, read_abbreviation_from_json

from doc.utils import cleanup_text, split_text, identify_tag, get_name, _change_to_regex_string


class Document(object):
    """Parsing HTML/XML/PDF file, preprocessing text, and tokenize."""
    __version__ = 2.0

    ELEMENT = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K",
               "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
               "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I",
               "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
               "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr",
               "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf",
               "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og", "Uue"]

    ELEMENT_NAMES = ["hydrogen", "helium", "lithium", "beryllium", "boron", "carbon", "nitrogen", "oxygen", "fluorine",
                     "neon", "sodium", "magnesium", "aluminium", "silicon", "phosphorus", "sulfur", "chlorine", "argon",
                     "potassium", "calcium", "scandium", "titanium", "vanadium", "chromium", "manganese", "iron",
                     "cobalt", "nickel", "copper", "zinc", "gallium", "germanium", "arsenic", "selenium", "bromine",
                     "krypton", "rubidium", "strontium", "yttrium", "zirconium", "niobium", "molybdenum", "technetium",
                     "ruthenium", "rhodium", "palladium", "silver", "cadmium", "indium", "tin", "antimony", "tellurium",
                     "iodine", "xenon", "cesium", "barium", "lanthanum", "cerium", "praseodymium", "neodymium",
                     "promethium", "samarium", "europium", "gadolinium", "terbium", "dysprosium", "holmium", "erbium",
                     "thulium", "ytterbium", "lutetium", "hafnium", "tantalum", "tungsten", "rhenium", "osmium",
                     "iridium", "platinum", "gold", "mercury", "thallium", "lead", "bismuth", "polonium", "astatine",
                     "radon", "francium", "radium", "actinium", "thorium", "protactinium", "uranium", "neptunium",
                     "plutonium", "americium", "curium", "berkelium", "californium", "einsteinium", "fermium",
                     "mendelevium", "nobelium", "lawrencium", "rutherfordium", "dubnium", "seaborgium", "bohrium",
                     "hassium", "meitnerium", "darmstadtium", "roentgenium", "copernicium", "nihonium", "flerovium",
                     "moscovium", "livermorium", "tennessine", "oganesson", "ununennium"]

    MOLECULAR_NAMES = ["carbon dioxide", "carbon oxide", 'methane', 'methanol', 'ethanol']

    MOLECULAR = ["CO2", "CO", "NH3", "H2O2", "H2O", "CO3", "ClO3-", "KOH", "Cl2", "Cl", "Br2", "Br", "CH4", 'C2H5OH',
                 'HCl', "NO2", "CaCO3", "NaCl", r"Ca\(OH\)2"]

    def __init__(self, filepath, reader=None, **database):
        """
        :param filepath: (str, bytes, os.PathLike or pathlib.Path) Path of file
        :param reader: [CDEPdfParser, CDEHtmlParser, CDEXmlParser, GeneralXmlParser, ElsevierXmlParser]
        :param database: -
        """
        path = Path(filepath)
        self.filepath = path
        self.suffix = path.suffix

        self.ELEMENTS_AND_NAMES = self.MOLECULAR + self.MOLECULAR_NAMES + self.ELEMENT + self.ELEMENT_NAMES + [
            en.capitalize() for en in self.ELEMENT_NAMES + self.MOLECULAR_NAMES]

        if not path.exists():
            raise FileNotFoundError()
        if self.suffix not in ['.html', '.xml', '.pdf']:
            raise TypeError()

        if reader:
            self.reader = reader
        else:
            self.reader = Default_readers[self.suffix]

        self.elements = self._parse()
        self.metadata = self._get_metadata()

        self.database = {}
        self.set_database(**database)

        self._strange = self._find_advanced_chemical()

    def set_database(self, **kwargs):
        abs_path = Path(__file__).parent
        database_list = ['chemical', 'small_molecule', 'ion', 'unit', 'abbreviation']
        for key in kwargs.keys():
            if key not in database_list:
                raise AttributeError('Database must be chemical, small_molecule, ion, unit, and abbreviation')

        if 'chemical' in kwargs:
            self.database['chemical'] = kwargs['chemical']
        else:
            self.database['chemical'] = DataStorage('chemical', 'c')

        if 'small_molecule' in kwargs:
            self.database['small_molecule'] = kwargs['small_molecule']
        else:
            self.database['small_molecule'] = DataStorage('small_molecule', 's')

        if 'ion' in kwargs:
            self.database['ion'] = kwargs['ion']
        else:
            self.database['ion'] = DataStorage('ion', 'i')

        if 'unit' in kwargs:
            self.database['unit'] = kwargs['unit']
        else:
            self.database['unit'] = UnitStorage()

        if 'abbreviation' in kwargs:
            self.database['abbreviation'] = kwargs['abbreviation']
        else:
            self.database['abbreviation'] = read_abbreviation_from_json(abs_path / 'storage/database/abb_list.json')

    def _parse(self):
        if not self.reader.check_suffix(self.suffix):
            raise TypeError(f'expected {self.suffix} type reader, not {self.reader.suffix} type reader')
        return self.reader.parsing(self.filepath)

    def _get_metadata(self):
        return self.reader.get_metadata(self.filepath)

    @staticmethod
    def identify_tag(label: str):
        return identify_tag(label)

    def get_name(self, hashtag):
        """
        find original word of hashtag
        :param hashtag: (str) hash tag
        :return: (str) original word of hashtag
        """
        return get_name(hashtag, self.database)

    def find_unit_from_text(self, text):
        unit_sub = self.MOLECULAR + self.MOLECULAR_NAMES
        return self.database['unit'].find_unit(text, unit_sub)

    def _find_advanced_chemical(self, list_of_text=None, cut_off=3):
        strange_set = set()
        strange_counter = Counter()

        if not isinstance(list_of_text, list):
            list_of_text = self.elements

        text_list = [cleanup_text(para) for para in list_of_text]
        for text in text_list:
            split = split_text(text)
            word_strange = filter(lambda t: not regex.search(
                r"^(?:(?:[±×+.-]|\d)+|\d+(st|rd|nd|th)|-.+?-|.+_\[.+\].*|([Ff]igure|[Tt]able|[Ff]ig|ESI)\S+|[A-Z]?[a-z]+([-/][A-Z]?[a-z]+)+|[A-Z]?[a-z]+)$",
                t) and len(t) > 1, split)
            a1, a2 = tee(word_strange)
            strange_counter.update(chain.from_iterable(map(lambda token: [z.group("name") for z in regex.finditer(
                r"(?P<name>([A-Za-z0-9])*[A-Za-z](?(1)|[A-Za-z0-9])*)", token)], a2)))
            strange_set = strange_set.union(a1)

        for word in iter(strange_set.copy()):
            word_analysis = [z.group("name") for z in
                             regex.finditer(r"(?P<name>([A-Za-z0-9])*[A-Za-z](?(1)|[A-Za-z0-9])*)", word)]
            if not word_analysis:
                strange_set.remove(word)
            elif reduce(lambda a, t: a + strange_counter[t], word_analysis, 0) / len(word_analysis) < cut_off:
                strange_set.remove(word)

        return strange_set

    def _set_chemical_to_hash(self, chemical_name, set_advanced_chem=True, hashcode=None):
        chemical_name = regex.sub(r"\(bold\)", "", chemical_name)

        if chemical_name in ['estimated', 'trend', 'discovery', 'bore', 'and', 'end', ',', 'Table',
                             'retain', 'backbone', 'scene', 'lamps', 'zip', 'ribbon', 'school', 'join', 'cyan', 'theme',
                             'plugin',
                             'ramp', 'doc', 'constant', 'constance', 'visible-light', 'measuring', 'estimate']:
            return chemical_name

        elif chemical_name in self.database['chemical']:  # Already existed
            return self.database['chemical'][chemical_name]
        elif chemical_name in self.database['ion']:
            return self.database['ion'][chemical_name]
        elif chemical_name in self.database['small_molecule']:
            return self.database['small_molecule'][chemical_name]
        elif chemical_name in self.ELEMENT:
            return "-e{}-".format(str(self.ELEMENT.index(chemical_name)).zfill(6))
        elif set_advanced_chem and chemical_name in self._strange:
            return "-strange({})-".format(chemical_name)
        else:
            hashcode = self.database['chemical'].append(chemical_name, )
            self.ELEMENTS_AND_NAMES = _change_to_regex_string([chemical_name], self.ELEMENTS_AND_NAMES)
            return hashcode

    def _save_abbreviation(self, abb_tuple):
        abb_text1 = " ".join(abb_tuple[0])
        abb_text2 = " ".join(abb_tuple[1])
        abb_type = abb_tuple[2]

        if len(abb_tuple[0]) == 1:
            if len(abb_tuple[1]) == 1 and len(abb_text1) > len(abb_text2):
                abb_def = abb_text1
                abb_name = abb_text2
            else:
                abb_def = abb_text2
                abb_name = abb_text1

        elif len(abb_tuple[1]) == 1:
            abb_def = abb_text1
            abb_name = abb_text2

        else:
            return None

        # validation_check
        if len(regex.findall(r"\(|\{|\[", abb_def)) != len(regex.findall(r"\)|\]|\}", abb_def)):
            return None

        abb_front_char = reduce(lambda x, y: x + "".join(regex.findall(r"^\S|[^a-z]", y)),
                                regex.split(r",|\s|-", abb_def), "")

        if abb_name[-1] == 's' and abb_def[-1] == 's':
            abb_front_char += 's'
        abb_char = regex.sub(r"(,|\s|[-])", "", abb_name)
        ratio = fuzz.ratio(abb_char.lower(), abb_front_char.lower())

        if ratio < 70 and not abb_type:
            return None

        if regex.findall(r"(?i)reaction", abb_def):
            abb_type = None
        elif abb_type:
            if not Paragraph(abb_def.replace("-", " - ")).cems:
                abb_type = None
        elif regex.findall(r"(?<=^|\s|-)({})(?=\s|-|$)".format(r"|".join(self.ELEMENTS_AND_NAMES)), abb_def):
            abb_type = 'CM'

        abb_ = self.database['abbreviation'].append(abb_name, )

        if abb_type:
            hashcode = self._set_chemical_to_hash(abb_name, set_advanced_chem=False)
            self._set_chemical_to_hash(abb_def, set_advanced_chem=False, hashcode=hashcode)

        return abb_, abb_def, abb_type

    def _change_to_hash(self, split, set_advanced_chem):
        new_split = []

        for i, word in enumerate(split):
            tag = self.identify_tag(word)
            new_word = word

            if tag:
                pass

            elif word in self.ELEMENT:
                if i and split[i - 1] != "-end-":
                    new_word = "-e{}-".format(str(self.ELEMENT.index(word)).zfill(6))

            elif word in self.database['small_molecule']:
                chem = self.database['small_molecule'][word]
                new_word = chem

            elif word in self.database['ion']:
                chem = self.database['ion'][word]
                new_word = chem

            elif word in self.database['chemical']:
                chem = self.database['chemical'][word]
                new_word = chem

            elif word in self.database['abbreviation']:
                abb = self.database['abbreviation'].get_abbreviation(word)
                abb_name = self.database['abbreviation'].get_name(word)
                if abb.ABB_type == 'CM':
                    new_word = self._set_chemical_to_hash(word, set_advanced_chem=False)

            else:
                num_string = ['several', 'few', 'first', 'second', 'third', 'forth', 'fifth', 'ninth', 'twelfth', 'one',
                              'two',
                              'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve',
                              'thirteen',
                              'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty', 'thirty',
                              'forty', 'fifty', 'sixty', 'seventy', 'ninety', 'hundred']
                num_string = sorted(num_string, reverse=True)
                num_string = r"|".join(num_string)
                num_match = regex.match(
                    fr"^(?P<num>[±×+-]?\s?(\d+\.\d+|\d+/\d+|\d+|{num_string})(st|nd|rd|th)?(?:\(\d+\))?)$",
                    word)
                chemical_match = Paragraph(word.replace("-", " - ")).cems

                if num_match:
                    new_word = "-num({})-".format(num_match.group("num"))

                elif chemical_match:
                    if regex.findall(r"^(?:(?:{element})\d*)+[+-]$".format(element=r"|".join(self.ELEMENT)), word):
                        new_word = self.database['ion'].append(word, )

                    elif regex.findall(
                            r"^(?:(?:{element})|{molecular})\d*(?:\b|\s?[(][\u2160-\u217FIXVixv]{iter}[)]\B)$".format(
                                element=r"|".join(self.ELEMENT), molecular=r"|".join(self.MOLECULAR), iter=r"{1,5}"),
                            word):
                        new_word = self.database['small_molecule'].append(word, )

                    elif regex.findall(r"^(?i)({})(s|es)?$".format(r"|".join(self.ELEMENT_NAMES)), word):
                        new_word = self.database['small_molecule'].append(word, )

                    else:
                        new_word = self._set_chemical_to_hash(word, set_advanced_chem=True)

                else:  # Not chemical / num match
                    if word in self.database['abbreviation']:
                        pass

                    elif set_advanced_chem and word in self._strange:
                        word_remove_bold = regex.sub(r"\(bold\)", "", word)
                        new_word = "-strange({})-".format(word_remove_bold)

                        self.ELEMENTS_AND_NAMES = _change_to_regex_string([word], self.ELEMENTS_AND_NAMES)

            new_split.append(new_word)

        assert len(split) == len(new_split)
        return new_split

    def _merge_token(self, split):
        activation = False
        for i, word in enumerate(split):
            if not word:
                continue

            tag = self.identify_tag(word)
            chemical_tf = tag in ['chemical', 'element', 'small_molecule', 'ion', 'strange']

            if not activation:
                if chemical_tf:
                    activation = True
                    if not i:
                        continue
                    try:
                        before_word = self.get_name(split[i - 1])
                    except IndexError:
                        before_word = None

                    # Prefix of chemical such as 'modified'
                    if isinstance(before_word, str):
                        if before_word in ['metal', "poly", 'pure', 'bare', 'pristine', 'exfoliate'] or \
                                regex.match(r"\S+(ed|ic)$", before_word):
                            chem = "{} {}".format(before_word, self.get_name(split[i]))
                            split[i] = chem
                            split[i - 1] = None

                elif regex.match("^[\u2160-\u217FIXVixv]{1,5}$", word):
                    if split[i - 1] == "(" and \
                            self.identify_tag(split[i - 2]) in ['chemical', 'element', 'small_molecule', 'ion'] and \
                            split[i + 1] == ")":
                        chem = self.get_name(split[i - 2]) + " " + split[i - 1] + self.get_name(split[i]) + split[i + 1]
                        split[i - 2], split[i - 1], split[i] = None, None, None
                        hashcode = self._set_chemical_to_hash(chem)
                        split[i + 1] = hashcode

                elif tag == 'number':
                    num_tree = self._num_tree()
                    child_tree = num_tree
                    try:
                        if self.identify_tag(split[i - 1]) == 'number':
                            split[i] = "-num({} {})-".format(self.get_name(split[i - 1]), self.get_name(split[i]))
                            split[i - 1] = None
                        elif split[i - 1] in ['to', '-'] and self.identify_tag(split[i - 2]) == 'number':
                            split[i] = "-num({})-".format(
                                self.get_name(split[i - 2]) + " " + split[i - 1] + " " + self.get_name(split[i]))
                            split[i - 2], split[i - 1] = None, None

                            if split[i + 1] in ['and', 'or'] and self.identify_tag(split[i + 2]) == 'number' and \
                                    split[i + 3] not in ['to', '-']:
                                num = self.get_name(split[i]) + " " + split[i + 1] + " " + self.get_name(split[i + 2])
                                split[i] = f"-num({num})-"
                                split[i + 1], split[i + 2] = None, None
                    except IndexError:
                        pass
                    else:
                        continue

                    iter_num = 1
                    word = split[i - iter_num]

                    while word in child_tree:
                        child_tree = child_tree[word]
                        iter_num += 1
                        word = split[i - iter_num]

                    if child_tree[None]:
                        revised_word = ""
                        for iter_ in range(1, iter_num):
                            revised_word = split[i - iter_] + " " + revised_word
                            split[i - iter_] = None
                        revised_word += self.get_name(split[i])
                        split[i] = "-num({})-".format(revised_word)
                else:
                    continue

            # Activation_case
            elif not chemical_tf:
                # Suffix of chemical
                suffix_list = r"|".join(
                    ['acid', "'", 'oxide', 'dioxide', 'monoxide', 'trioxide', 'powder', 'crystal', 'crystalline',
                     'particle', 'hollow',
                     'sphere', 'film', 'web', 'sheet', 'flower', 'fiber', 'atom', 'pore', 'composite', 'N[A-Z]s',
                     'metal'])

                sub_chemical_suffix_list = ['in', 'on', 'based', 'layered', 'coated', 'with', 'decorated', 'activated',
                                            'deposited', 'combined', 'supported']

                if regex.match(fr"(?i)(ion|anion|cation)(s|es|ies)?", word):
                    if self.identify_tag(split[i - 1]) == 'ion':
                        split[i] = split[i - 1]
                        split[i - 1] = None
                    else:
                        ion_name = self.get_name(split[i - 1]) + " " + self.get_name(split[i])
                        hashtag = self.database['ion'].append(ion_name, )
                        split[i] = hashtag
                        split[i - 1] = None

                elif regex.match(fr"({suffix_list})(s|es|ies)?", word) or \
                        regex.match(r"^(nano|micro|macro|meso)\S+", word):
                    split[i] = self.get_name(split[i - 1]) + " " + self.get_name(split[i])
                    split[i - 1] = None

                elif word in sub_chemical_suffix_list:
                    try:
                        next_word = split[i + 1]
                    except IndexError:
                        next_word = None

                    if next_word in sub_chemical_suffix_list or \
                            self.identify_tag(next_word) in ['chemical', 'element', 'small_molecule', 'ion', 'strange']:
                        split[i] = self.get_name(split[i - 1]) + " " + self.get_name(split[i])
                        split[i - 1] = None
                    else:
                        activation = False
                        chem_name = self.get_name(split[i - 1])
                        hashcode = self._set_chemical_to_hash(chem_name)
                        split[i - 1] = hashcode
                else:
                    activation = False
                    chem_name = self.get_name(split[i - 1])
                    hashcode = self._set_chemical_to_hash(chem_name)
                    split[i - 1] = hashcode

            else:
                split[i] = self.get_name(split[i - 1]) + " " + self.get_name(split[i])
                split[i - 1] = None

        if activation and split[-1]:  # Remove last term
            # if not bar_activation and not slash_activation:
            chem_name = self.get_name(split[-1])
            hashcode = self._set_chemical_to_hash(chem_name)
            split[-1] = hashcode

        return [token for token in split if token != " "]

    def find_abbreviation_from_text(self, text, return_new_abbreviation=False):
        def abb_sub(group):
            abb_def = group.group("ABB")
            change_word = self.database['abbreviation'].get_name(abb_def)
            if change_word:
                return change_word

            change_word = self.database['abbreviation'].get_name(abb_def.lower())
            if change_word:
                return change_word

            return abb_def

        abb_regex = self.database['abbreviation'].abb_regex
        text = regex.sub(fr"(?<=^|\s)(?P<ABB>{abb_regex})(?=\s|$)", abb_sub, text)

        sentence = Paragraph(text)

        # Find (ABB)
        abb_list = sentence.abbreviation_definitions
        new_abb_list = []
        for abb_tuple in abb_list:
            abb_output = self._save_abbreviation(abb_tuple)
            if abb_output:
                new_abb_list.append(abb_output)

        if return_new_abbreviation:
            return text, new_abb_list
        else:
            return text

    @staticmethod
    def _num_tree():
        def tree_maker(iterator, tree=None):
            if not isinstance(tree, dict):
                tree = {None: False}

            if not isinstance(iterator, str):
                for node in iterator:
                    tree = tree_maker(node, tree)
                return tree
            else:
                split_iter = iterator.split()
                iterator = [None] + split_iter
                iterator.reverse()

                child_tree = tree
                for word in iterator:
                    if not word:
                        child_tree[word] = True

                    elif word in child_tree:
                        child_tree = child_tree[word]
                    else:
                        child_tree[word] = {None: False}
                        child_tree = child_tree[word]
                return tree

        wordlist = ["about", "ca .", "ca.", "less than", "more than", "approximately", "around", "roughly", "up to",
                    "nearly", "~", "over",
                    "average", "equal to", "only", "great than", "close to", "correspond to", "twice", "only ~",
                    "maximum at", "maximum",
                    "estimate to be", "below", "in range of", "almost", "only about", "at around" "maximum of",
                    "minimum of", 'close to', 'estimate to be',
                    'decrease to', 'increase to', 'calculate to be', 'high than that of', 'small than', 'as high as',
                    'slightly', 'circa', 'measure to be',
                    'large than', 'only', 'decrease by', 'observe at', 'decrease from', 'range between', 'relative to',
                    'increase from', 'increase to',
                    'as determine by', 'large than that of', 'up to ~', 'vary between', 'amount to', 'on average',
                    'value of', 'equal', 'equal at',
                    'at least', 'measure at', 'approximate', 'equivalent to', 'increase to', 'larger than',
                    'constant at', 'center at', 'determine to be',
                    'slightly', 'as low as', 'slightly high', 'approximate', 'high than that of', 'in range from',
                    'vary from', 'low than', 'above',
                    '>', '<', 'no more than', 'no less than', 'no greater than', 'not increase to', 'not decrease to',
                    'no large than', 'greater than', ]

        return tree_maker(wordlist)

    def tokenize(self, sentence, cut_off=False, set_advanced_chem=True):
        if isinstance(sentence, str):
            sentence = Paragraph(sentence)

        text = cleanup_text(sentence)

        if not text:
            return None

        if cut_off:
            if len(sentence.sentences) < 3:
                return None
            if len(regex.compile(r"[äüö]").findall(text)) > 3:
                return None
        # find ABB
        # text, new_abb_list = self.find_abbreviation_from_text(text, return_new_abbreviation=True)
        _, new_abb_list = self.find_abbreviation_from_text(text, return_new_abbreviation=True)

        # find unit
        text = self.find_unit_from_text(text)

        split = split_text(text, concat_bracket=True)

        # class-1 classify
        split = self._change_to_hash(split, set_advanced_chem)

        # find abbreviation
        for abb, abb_def, abb_type in new_abb_list:
            if abb_type:
                continue

            new_abb_type = regex.findall(r"(?<=^|\s|-)({})(?=\s|-|$)".format(r"|".join(self.ELEMENTS_AND_NAMES)),
                                         abb_def)

            if not new_abb_type or regex.findall(r"(?i)reaction", abb_def):
                pass

            else:
                abb.change_abb_type(abb_def, 'CM')
                abb_name = abb.ABB_name
                hash_code = self.database['chemical'].get(abb_def, None)
                abb_cem = self._set_chemical_to_hash(abb_name, set_advanced_chem=False, hashcode=hash_code)
                for i, word in enumerate(split):
                    if word == abb_name:
                        split[i] = abb_cem

        merge_split = self._merge_token(split)

        for i, word in enumerate(merge_split):
            tag = self.identify_tag(word)
            if tag == 'chemical' or tag == 'strange':
                name = self.get_name(word)

                if " " in name:  # multi-token chemical
                    chem_analyze = self.regularize_chemical(name, tag).get('chemical', None)
                else:  # single-token chemical
                    chem_analyze = self._is_valid_chemical(name, tag)
                if not chem_analyze:
                    merge_split[i] = name

        merge_split = [word for word in merge_split if word is not None]

        return merge_split

    def regularize_chemical(self, chem_name, tag='strange'):

        suffix_list = r"|".join(
            ['powder', 'crystal', 'crystalline', 'particle', 'hollow', 'sphere', 'film', 'web', 'sheet', 'flower',
             'fiber',
             'atom', 'pore', 'composite', 'N[A-Z]s'])

        abb_regex = self.database['abbreviation'].abb_regex

        chem_name_revise = regex.sub(fr"(?P<ABB_name>{abb_regex})",
                                     lambda t: self.database["abbreviation"].get_name(t.group("ABB_name")),
                                     chem_name)

        end_group = r"|".join(
            ['type', 'based', 'layered', 'coated', 'decorated', 'activated', 'deposited', 'combined', 'supported',
             'ion', 'cation', 'anion', "N[A-Z]s", 'metal', 'metallic', 'bimetallic'])

        front_group = r"|".join(
            ['pristine', 'bare', 'bared', 'pure', 'modified', 'stabilized', '1D', '2D', '3D', '0D', '1-D', '2-D', '3-D',
             '0-D', 'metal', 'metallic', 'bimetallic'])

        chem_name_revise = regex.sub(
            fr"(?<=\S+)-(?P<pos>{end_group}|{suffix_list}|\S+ed)s?(?=\s|$)|(?<=^|\s)(?P<pos>{front_group}|\S+ed)-(?=\S+)",
            lambda t: " {} ".format(t.group("pos")), chem_name_revise)

        chem_iter = chem_name_revise.split()

        output = {'chemical': [''], 'chem_type': []}

        for chem in chem_iter:

            if chem in ['in', 'on', 'to', 'with']:
                output['chemical'].append('')

            elif chem in ['poly']:
                output['chemical'][-1] += f" {chem}"

            elif regex.match(fr"({end_group}|{front_group})", chem):

                output['chemical'].append('')
                output['chem_type'].append(chem)

            elif regex.match(fr"({suffix_list})(s|es|ies)?", chem) or regex.match(r"^(nano|micro|macro|meso)\S+", chem):
                output['chemical'].append('')
                output['chem_type'].append(chem)

            elif not self._is_valid_chemical(chem, tag, middle_possible=True):
                output['chemical'].append('')
                output['chem_type'].append(chem)

            elif regex.match(r".+[@/]", chem):
                chem_split = regex.sub(r"(?=.+)((?P<split>@)|\d/\d|(?P<split>/))",
                                       lambda t: " " if t.group("split") else t.group(0), chem).split()
                if len(chem_split) - 1:
                    for chem_s in chem_split:
                        output['chemical'].append('')
                        output['chemical'][-1] += f' {chem_s}'
                else:
                    output['chemical'][-1] += f' {chem}'
            else:
                output['chemical'][-1] += f" {chem}"

        output['chemical'] = [word.strip() for word in output['chemical'] if word]

        return output

    def _is_valid_chemical(self, name, tag='strange', middle_possible=False):
        if name not in self.database['chemical'] and (name not in self._strange) and \
                name not in self.database['ion'] and name not in self.database['small_molecule']:
            if name in self.database['abbreviation'] and self.database['abbreviation'].get_abbreviation(name).ABB_type:
                pass
            else:
                return False

        if regex.search(r"_\[.+\]", name):
            return False

        elif not middle_possible and regex.findall(r"(yl|lic)$", name):
            return False

        elif regex.search(r"(ing|ed)$", name):  # Alkyl, caboxylic, N2-sorting, N2-sorpted
            return False

        elif regex.match(r"^(?i){}(s|es)$".format(r"|".join(self.MOLECULAR_NAMES)), name):

            return True

        elif not middle_possible and self._remove_not_chem_group(name):
            return False

        elif regex.match(r"^([A-Za-z]/[A-Za-z])0?$", name):  # P/P0, Z/Z0
            return False

        elif regex.search(r"[Ff]igure|[Tt]able|[Ff]ig|ESI", name):  # Figure2
            return False

        elif regex.search(r"\S+[-](type|fold|storage)(s|es)?", name):  # N2-type / 3D-fold / Li-storage
            return False

        elif tag == 'strange':

            if regex.match(r"^[A-Za-z][a-z]*[.]$", name):  # Word. / word.
                return False

            elif regex.match(r"[A-Za-z][a-z]*([-/\s][A-Za-z][a-z]*)+$", name):
                activation_s = False
                split_words = regex.split(r"[/-]", name)
                for split_word in split_words:
                    if split_word in self.database['chemical'] or split_word in self.database['small_molecule'] or \
                            split_word in self.ELEMENTS_AND_NAMES:
                        activation_s = True
                if activation_s:
                    return True
                else:
                    return False

            elif self.database['unit'].is_unit(name):
                return False

        return True

    @staticmethod
    def _remove_not_chem_group(word):
        functional = ["alkane", 'alkene', 'alkyne', 'haloalkane', 'fluoroalkane', 'chloroalkane', 'bromoalkane',
                      'iodoalkane',
                      'alcohol', 'ketone', 'aldehyde', 'acyl halide',
                      'carbonate', 'carboxylate', 'carboxylic acid', 'ester', 'methoxy', 'hydroperoxide',
                      'peroxide', 'ether', 'hemi-?acetal', 'hemi-?ketal', 'acetal', 'ketal', 'orthoester',
                      'heterocycle', 'orthocarbonate ester', 'organic acid anhydride', 'amide', 'amine',
                      'imine', 'imide', 'azide', 'azo compound', 'cyanate', 'isocyanate', 'nitrate', 'nitrile',
                      'nitro compound', 'nitroso compound', 'oxime', 'carbamate', 'carbamate ester',
                      'thiol', 'sulfide', 'thioether', 'disulfide', 'sulfoxide', 'sulfone', 'surfinic acid',
                      'surfonic acid', 'sulfonate ester', 'thiocyanate', 'thioketone', 'thial', 'thiocarboxylic acid',
                      'thioester', 'dithiocarboxylic acid', 'dithiocarboxylic acid ester', 'phosphine',
                      'phosphonic acid', 'phosphate', 'phosphodiester', 'boronic acid', 'boronic ester',
                      'borinic acid', 'borinic ester', 'alkyllithium', 'alkylmagnesium halide', 'alkylaluminium',
                      'silyl ether', 'oxide', 'carbide', 'metal', 'semi-metal', 'carbon nitride', 'enol']

        not_chemical = ["micropore", 'macropore', 'meso', 'micro', 'macro', 'mesopore', 'estimated', 'trend',
                        'discovery', 'bore',
                        'retain', 'backbone', 'scene', 'lamps', 'zip', 'ribbon', 'school', 'join', 'cyan', 'theme',
                        'plugin',
                        'ramp', 'doc', 'constant', 'constance', 'poly', 'polymer', 'estimate', 'pure', 'bare',
                        'pristine',
                        'exfoliate', 'polymeric', 'visible-light', 'metal', '1D', '2D', '3D', 'measuring', 'pH',
                        'reforming']

        return regex.match(
            r"(?i)^(?:{func}|{nocem})(?:s|es)?$".format(func=r"|".join(functional), nocem=r"|".join(not_chemical)),
            word)

    @property
    def title(self):
        return self.metadata['title']

    @property
    def doi(self):
        return self.metadata['doi']

    @property
    def journal(self):
        return self.metadata['journal']

    @property
    def author_list(self):
        return self.metadata['author_list']

    @property
    def date(self):
        return self.metadata['date']

    def __repr__(self):
        return f"""Title : {self.title}\nDoi : {self.doi}\nJournal : {self.journal}\nDate : {self.date}"""
