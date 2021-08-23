import numpy as np
from copy import deepcopy
import regex
import json
import warnings

from ase.formula import Formula
from tensorflow.keras.preprocessing.sequence import pad_sequences

from doc.storage import UnitStorage, DataStorage
from doc.utils import split_text, cleanup_text
from libs.property_extractor.extract_paragraph_v2 import matching_algorithm, word_nearest, Value, Word
from error import DatabaseError, MerError, MofError
from utils import transform_unit
from mer import material_entity_recognition


class MOF(object):
    __version__ = '2.1.0'

    metal_symbols = ("Li", "Be", "Mg", "Al", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga",
                     "Rb", "Sr", "Y", "Ge", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Cs",
                     "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
                     "Hf", "Pt", "Au", "Hg", "Pb", "Ln", "Xe", "Ta", "W", "Re", "Os", "Ir", "Tl", "Bi", "Po", "Fr",
                     "Ra", "Ac", "Th", "Pa",  "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", 'U',
                     'Ca', 'Sb')

    metal_names = (
        "Lithium", "Beryllium", "Magnesium",  # "Sodium"
        "Aluminum", "Calsium", "Scandium", "Potassium",
        "Titanium", "Vanadium", "Chromium", "Manganese", "Iron",
        "Cobalt", "Nickel", "Copper", "Zinc", "Gallium",
        "Rubidium", "Strontium", "Yttrium", "Germanium"
                                            "Zirconium", "Niobium", "Molybdenum", "Technetium", "Ruthenium",
        "Rhodium", "Palladium", "Silver", "Cadmium", "Indium",
        "Caesium", "Barium",  # "Tin"
        "Lanthanum", "Cerium", "Praseodymium", "Neodymium", "Promethium",
        "Samarium", "Europium", "Gadolinium", "Terbium", "Dysprosium",
        "Holmium", "Erbium", "Thulium", "Ytterbium", "Lutetium",
        "Hafnium", "Platinum", "Silver", "Uranium",
        "Mercury", "Lead", "Lanthanide", "Gadolinium",
    )

    solvent_names = (
        'Acetate', 'Benzene', 'Ethyl acetate', 'Pentane', 'Formic', 'Toluene', 'Methanol',
        'Tetrahydrofuran', 'Acid', '1,4-Dioxane', 'Diethyl ether', 'Chloroform', 'Ammonia',
        'Dichloromethane', 'Acetone', 'Tetrachloride', 'Isopropyl alcohol', 'Water',
        'Dimethyl sulfoxide', 'Propylene carbonate', 'Formic acid', 'Isopropyl', 'Cyclohexane',
        'Cyclopentane', 'Acetonitrile', 'Carbon tetrachloride', 'Nitromethane', 'Hexane',
        'Propanol', 'Butanol', 'Propylene', 'Acetic acid', 'Dimethylformamide', 'Ethanol', 'Alcohol',
        'phenanthroline', 'solution', 'H3PO4', 'ethylenediamine',
        'elthylenediamine', 'Urea', 'aqua', 'metano', 'heptane')

    solvent_symbols = ('IPA', 'MeOH', 'DCM', 'THF', 'DMA', 'DMF', 'DMSO', 'DMI', 'H2O', "CH2Cl", "CH3CN", "CH3OH",
                       "MeOH", "NaOH", "DEF", 'EtOH', 'NaN3', 'TEA', 'KOH', 'HNO3', 'HCl', 'HF', 'Et3N',
                       'CHCl3', 'KSCN', 'CH3COOH', 'Et2O', 'C2H5OH', 'NH4SCN', 'NH3', 'KI', 'H3PO3',
                       'NaN(CN)2', 'NH3', 'Net3', 'PPh3', 'NH4CL', 'BKR', 'KNCSe', 'H3BO3', 'H2SO4', 'C6H5COONH4',
                       'KCl', 'NaHCO3', 'kpf6', 'NaF', 'HBF4', 'NaSCN', 'SeO2', 'NaNCS', 'DMSO', 'MeCN',
                       'Na2CO3', 'KNCS',)

    def __init__(self, **kwargs):
        """
        Text mining toolkit for Metal organic framework synthesis.
        Using Classmethod <MOF.from_paragraph> to create MOF from paragraph

        :param kwargs: parameter of MOF
        >> name : name of MOF
        """
        self.name = kwargs.get("name")
        self.symbol = kwargs.get("symbol")
        self.doi = kwargs.get('doi')
        self._text = kwargs.get('text')
        self.operation = kwargs.get('operation', [])
        self.property = kwargs.get('prop')
        self.method = kwargs.get('method')
        self._convert_precursor = kwargs.get('convert_precursor', False)
        self.metadata = kwargs.get('metadata')

        if 'M_precursor' in kwargs or 'O_precursor' in kwargs or 'S_precursor' in kwargs:
            self.M_precursor = kwargs.get('M_precursor', [])
            self.O_precursor = kwargs.get('O_precursor', [])
            self.S_precursor = kwargs.get('S_precursor', [])
            self._precursor = self.M_precursor + self.O_precursor
            self._etc = self.S_precursor
            self._target = [self.name, self.symbol]
            # self._get_simple_metal_name()

        else:
            self.M_precursor = []
            self.O_precursor = []
            self.S_precursor = []
            self._precursor = kwargs.get('precursor', [])
            self._target = kwargs.get('target', [])
            self._etc = kwargs.get('etc', [])

        if kwargs.get('standard_unit', True):
            self.time = transform_unit(kwargs.get('time'), return_type='dict')
            self.temperature = transform_unit(kwargs.get("temperature"), return_type='dict')
        else:
            self.time = kwargs.get('time')
            self.temperature = kwargs.get('temperature')

    def __repr__(self):
        if self.symbol:
            return f"{self.name} ({self.symbol})"

        return f"{self.name}"

    def __getitem__(self, item):
        dict_ = self.to_dict()
        return dict_[item]

    @classmethod
    def from_paragraph(cls, paragraph, database=None, standard_unit=True, classify_material=True,
                       convert_precursor=False, character_embedding=True, **kwargs):
        """
        paragraph : (str) MOF synthesis paragraph
        database : (dict) dictionary of <doc.storage.DataStorage>. (default : None)
        character_embedding : (bool) If False, character embedding is not used in material entity recognition.
                              (default = True)

        **kwargs:
        metadata : (dict or defaultdict) dictionary of title, doi, journal, date, and author list
        """
        if database is None:
            database = {}
        elif not isinstance(database, dict):
            raise DatabaseError('database must be dictionary of DataStorage, UnitStorage')

        if 'unit' not in database:
            database['unit'] = UnitStorage()
        if 'chemical' not in database:
            database['chemical'] = DataStorage('chemical', 'c')

        # material entity recognition
        method = _get_method(paragraph)
        token_sents, bio_tags = material_entity_recognition(paragraph, character_embedding=character_embedding)

        # Get materials from tokens
        materials = _get_materials_from_tokens(token_sents, bio_tags)
        symbol = materials.get('symbol')
        targets = materials.get('targets')
        precursor_names = materials.get('precursors')
        etc_names = materials.get('etc')
        try:
            name = targets[0]
        except IndexError:
            name = None

        # Get condition from tokens
        condition, extensive, reaction, operation = _get_condition(token_sents, bio_tags, database)

        precursor = []
        for precursor_name in precursor_names:
            composition = extensive.get(precursor_name, [])
            precursor.append({'name': precursor_name, 'composition': composition})

        etc = []
        for etc_name in etc_names:
            composition = extensive.get(etc_name, [])
            etc.append({'name': etc_name, 'composition': composition})

        time = condition.get('Time')
        temperature = condition.get("Temperature")
        ph = condition.get("pH")

        if not precursor and not etc and not targets:
            raise MofError('There are no materials in paragraph')

        # Generate MOF
        mof = MOF(name=name, symbol=symbol, precursor=precursor, time=time, temperature=temperature, pH=ph,
                  standard_unit=standard_unit, target=targets, etc=etc, text=paragraph, operation=operation,
                  prop=reaction, convert_precursor=convert_precursor, method=method, **kwargs)

        if classify_material:
            mof.classify_material()

        return mof

    @classmethod
    def from_dict(cls, file):
        if not isinstance(file, dict):
            raise TypeError()

        name = file.get('name')
        symbol = file.get('symbol')
        m_precursor = file.get('M_precursor')
        o_precursor = file.get('O_precursor')
        s_precursor = file.get('S_precursor')
        doi = file.get('doi')
        temperature = file.get('temperature')
        time = file.get('time')
        operation = file.get('operation')
        prop = file.get('property')
        text = file.get('text')
        method = file.get('method')

        return MOF(time=time, temperature=temperature, M_precursor=m_precursor, S_precursor=s_precursor,
                   O_precursor=o_precursor, doi=doi, name=name, symbol=symbol, operation=operation,
                   property=prop, text=text, method=method)

    @classmethod
    def from_json(cls, json_name):
        with open(json_name, 'r', encoding='utf-8') as f:
            file = json.load(f)
        return MOF.from_dict(file)

    @property
    def mo_ratio(self):
        if len(self.M_precursor) == 1 and len(self.O_precursor) == 1:
            m_comp = self.M_precursor[0]['composition']
            o_comp = self.O_precursor[0]['composition']
            m_mol = None
            o_mol = None
            for value, unit in m_comp:
                try:
                    if unit == 'mmol1.0':
                        m_mol = float(value)
                        break
                    elif unit == 'mol1.0':
                        m_mol = float(value) / 1000
                        break
                except (ValueError, TypeError):
                    pass

            for value, unit in o_comp:
                try:
                    if unit == 'mmol1.0':
                        o_mol = float(value)
                        break
                    elif unit == 'mol1.0':
                        o_mol = float(value) / 1000
                        break
                except (ValueError, TypeError):
                    pass

            if m_mol is not None and o_mol is not None:
                try:
                    return m_mol/o_mol
                except ZeroDivisionError:
                    return None
        return None

    def to_dict(self, extract_all=False):
        if extract_all:
            return {'name': self.name, 'symbol': self.symbol, 'target': self._target,
                    'precursor': self._precursor, 'etc': self._etc, 'M_precursor': self.M_precursor,
                    'O_precursor': self.O_precursor, 'S_precursor': self.S_precursor,
                    'MOratio': self.mo_ratio, 'temperature': self.temperature, 'time': self.time,
                    'operation': self.operation, 'property': self.property, 'method': self.method,
                    'doi': self.doi, 'text': self._text, 'metadata': self.metadata}
        else:
            return {'name': self.name, 'symbol': self.symbol, 'M_precursor': self.M_precursor,
                    'O_precursor': self.O_precursor, 'S_precursor': self.S_precursor,
                    'MOratio': self.mo_ratio, 'temperature': self.temperature, 'time': self.time,
                    'operation': self.operation, 'property': self.property, 'method': self.method,
                    'doi': self.doi, 'metadata': self.metadata}

    def append_material(self, material=None, astype='etc', *, name=None, composition=None, ):
        if astype not in ['precursor', 'target', 'etc']:
            raise ValueError()

        if not isinstance(material, dict):
            if composition is None:
                composition = []
            material = {'name': name, 'composition': composition}

        if astype == 'precursor':
            self._precursor.append(material)
        elif astype == 'target':
            self._target.append(material['name'])
        elif astype == 'etc':
            self._etc.append(material)

        return material

    def remove_material(self, material=None, *, name=None, composition=None):
        if not isinstance(material, dict) and composition is None:
            self._precursor = [precursor for precursor in self._precursor if precursor['name'] != name]
            self._target = [target for target in self._target if target != name]
            self._etc = [etc for etc in self._etc if etc['name'] != name]

        else:
            if not isinstance(material, dict):
                material = {'name': name, 'composition': composition}

            try:
                self._precursor.remove(material)
            except ValueError:
                pass
            try:
                self._target.remove(material['name'])
            except ValueError:
                pass
            try:
                self._etc.remove(material)
            except ValueError:
                pass

    def classify_material(self):
        metal_symbol = r"|".join(self.metal_symbols)
        symbol_regex = regex.compile(fr"({metal_symbol})[^a-z]")
        metal_name = r"|".join(self.metal_names)
        name_regex = regex.compile(fr"(?i)({metal_name})")

        # solvent_symbol = r"|".join(self.solvent_symbols)
        # solvent_symbol_regex = regex.compile(fr"({solvent_symbol})[^a-z]")
        # solvent_name = r"|".join(self.solvent_names)
        # solvent_name_regex = regex.compile(fr"(?i)({solvent_name})")

        self.M_precursor.clear()
        self.S_precursor.clear()
        self.O_precursor.clear()

        for material in self._precursor:
            name = material['name']
            if symbol_regex.search(name) or name_regex.search(name):
                self.M_precursor.append(material)
            else:
                self.O_precursor.append(material)

        """for material in self._etc:
            name = material['name']
            if solvent_name_regex.search(name) or solvent_symbol_regex.search(name):
                self.S_precursor.append(material)"""

        self.S_precursor = self._etc
        # self._get_simple_metal_name()

        if self._convert_precursor:
            self._convert_precursor_func()

    def get_material_list(self, attribute: str):
        """
        get material list from MOF
        :param attribute: attribute for MOF.
        :return: (list) name of materials
        """

        if attribute in ['M_precursor', 'O_precursor', 'S_precursor']:
            material_dict = getattr(self, attribute)
            material_list = [material['name'] for material in material_dict]
        elif attribute in ['precursor', 'etc']:
            material_dict = getattr(self, "_"+attribute)
            material_list = [material['name'] for material in material_dict]
        elif attribute in ['target']:
            material_list = getattr(self, "_"+attribute)
        else:
            raise KeyError("attribute must be 'M_precursor', 'O_precursor', 'S_precursor', 'target', 'precursor', "
                           "or 'etc'")

        return material_list

    def _get_simple_metal_name(self):
        p1 = regex.compile(r"·\S*")  # remove word after ·
        p2 = regex.compile(r"\W?[0-9]H2O")  # remove .xH2O
        p3 = regex.compile(r"\W?H2O")  # remove .H2O

        m_precursor = self.M_precursor
        if not m_precursor:
            return m_precursor

        for M_pre in m_precursor:
            if M_pre.get('simple_name'):
                continue

            m_pre_name = M_pre["name"]

            text = p1.sub('', m_pre_name)
            text = p2.sub('', text)
            text = p3.sub('', text)

            text = text.replace('[', '(')
            text = text.replace('<', '(')
            text = text.replace('{', '(')
            text = text.replace(']', ')')
            text = text.replace('>', ')')
            text = text.replace('}', ')')  # change to simple brackets (,)

            text = text.replace("Ac", "C2O2H3")  # add dictionary of chemical formula

            try:
                simple_m_pre_name = Formula(text, ).format('abc')
                if simple_m_pre_name:
                    M_pre['simple_name'] = simple_m_pre_name
            except Exception:
                pass

        return m_precursor

    def _convert_precursor_func(self):
        for metal_precursor in self.M_precursor:
            metal_name = metal_precursor['name']
            metal_formula = self._get_metal_precursor_formula(metal_name)
            if metal_formula:
                metal_precursor['formula'] = metal_formula

        for organic_precursor in self.O_precursor:
            organic_name = organic_precursor['name']
            organic_smiles = self._get_organic_precursor_smiles(organic_name)
            if organic_smiles:
                organic_precursor['smiles'] = organic_smiles

        for s_precursor in self.S_precursor:
            s_name = s_precursor['name']
            s_comp = self._get_solvent(s_name)
            if s_comp:
                s_precursor['solvent'] = s_comp

    def _get_metal_precursor_formula(self, m_pre_name):
        text = m_pre_name

        # check name vs formula
        p = regex.compile("|".join(map(lambda x: x.lower(), self.metal_names)))
        if p.findall(text.lower()):
            type_ = "chemical name"
        else:
            type_ = "chemical formula"

        if type_ == "chemical name":  # chemical formula

            # chemical name ex. manganese(II) perchlorate -> manganese perchlorate
            p0 = regex.compile(r"\(I*\)")
            text = p0.sub('', text)

            # read pre-made dictionary and convert
            dict_convert = json.load(open("./libs/precursor_convertor/m_pre_name_to_formula.json", "r"))
            if text in dict_convert.keys():
                return dict_convert[text]
            else:
                return None

        else:  # chemical formula

            # change to simple brackets ( or )
            text = text.replace("·", "$")
            text = text.replace('[', '(')
            text = text.replace('<', '(')
            text = text.replace('{', '(')
            text = text.replace(']', ')')
            text = text.replace('>', ')')
            text = text.replace('}', ')')
            text = text.strip()

            hydrate_regex = regex.compile(r"\W?(?P<num>[0-9]\.[0-9]|[0-9])?H2O$")
            hydrate = hydrate_regex.search(text)
            if hydrate:
                if hydrate.group('num') is None:
                    hydrate_num = '$H2O'
                else:
                    num_ = hydrate.group('num')
                    hydrate_num = f'${num_}H2O'
            else:
                hydrate_num = ""

            text = hydrate_regex.sub("", text)
            text = text.replace("Ac", "C2O2H3")  # add dictionary of chemical formula

            try:
                simple_m_pre_name = Formula(text).format('abc') + hydrate_num
                return simple_m_pre_name

            except Exception:
                return None

    def _get_organic_precursor_smiles(self, o_pre_name):
        dict_convert = json.load(open("./libs/precursor_convertor/o_pre_name_to_smiles.json", "r"))
        return dict_convert.get(o_pre_name)

    def _get_solvent(self, s_pre_name):
        dict_convert = json.load(open("./libs/precursor_convertor/s_pre_name.json", "r"))
        solvent_list = []
        for solvent, keyword in dict_convert.items():
            if regex.search(keyword, s_pre_name, regex.IGNORECASE):
                solvent_list.append(solvent)

        return solvent_list


def _get_consecutive_ids(data, stepsize=1):
    list_cons = np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)
    if len(list_cons[0]) == 0:
        return []
    else:
        return list_cons


def _get_materials_from_tokens(token_sents, bio_tags, maxlen=100):
    pad_token_sents = pad_sequences(token_sents, dtype="object", maxlen=maxlen, padding="post")

    target_names = []
    precursor_names = []
    etc_names = []
    symbol = None

    for pad_token_sent, bio_tag in zip(pad_token_sents, bio_tags):

        ids_targets = np.where(np.logical_or(bio_tag == 1, bio_tag == 2))[0]
        ids_precursors = np.where(np.logical_or(bio_tag == 3, bio_tag == 4))[0]
        ids_etcs = np.where(np.logical_or(bio_tag == 5, bio_tag == 6))[0]

        list_ids_targets = _get_consecutive_ids(ids_targets)
        list_ids_precursors = _get_consecutive_ids(ids_precursors)
        list_ids_etcs = _get_consecutive_ids(ids_etcs)
        # get target names
        for ids in list_ids_targets:
            target_name = " ".join(pad_token_sent[ids])
            # get symbol name
            if any(check in pad_token_sent[ids-1] for check in ["(", ","]):
                symbol = " ".join(pad_token_sent[ids])
            elif regex.search(r"(?<=\s)(1?[0-9]|[a-z]|[1-9][a-z])$", target_name):
                symbol = target_name.split()[-1]
                target_name = " ".join(target_name.split()[:-1])
                target_names.append(target_name)
            else:
                target_names.append(target_name)
        # get precursor names
        for ids in list_ids_precursors:
            precursor_name = " ".join(pad_token_sent[ids])
            precursor_names.append(precursor_name)

        # get etc names
        for ids in list_ids_etcs:
            etc_name = " ".join(pad_token_sent[ids])
            etc_names.append(etc_name)

    return {'targets': target_names, 'precursors': precursor_names, 'etc': etc_names, 'symbol': symbol}


def _get_hash_token(list_seq, list_mer, database):
    list_hash = []

    for seq, mer in zip(list_seq, list_mer):
        sentence = []
        for word, mer_label in zip(seq, mer):
            if not word:
                pass
            
            elif regex.search(r"^(FT-IR|IR|[Ss]pectr|\S*NMR)", word):  # remove words after IR / NMR
                return list_hash

            elif mer_label in [1, 3, 5]:
                word_hash = database['chemical'].append(word, )
                sentence.append(word_hash)
            elif mer_label in [2, 4, 6]:
                try:
                    last_chemical = sentence[-1]
                    last_original_name = database['chemical'][last_chemical]
                    new_chemical = f"{last_original_name} {word}"
                    word_hash = database['chemical'].append(new_chemical, )
                    sentence[-1] = word_hash
                    
                except (KeyError, DatabaseError):
                    warnings.warn('POS-I appear before POS-B appeared')
                    word_hash = database['chemical'].append(word, )
                    sentence.append(word_hash) 
            elif mer_label == 0:
                sentence.append(word)
            else:
                raise MerError(f'{mer_label} must be in 0-6')

        sentence = database['unit'].find_unit_from_list(sentence)
        list_hash.append(sentence)

    return list_hash


def _get_condition(list_seq, list_mer, database):  # Should be revise!
    list_hash = _get_hash_token(list_seq, list_mer, database)
    general_dictionary = {}
    reaction_dictionary = {}
    operation_list = []

    for sent in list_hash:
        chemical_type_dict = {}
        before_represent_chem = False

        new_sent, unit_dictionary, next_represent_chem = matching_algorithm(sent, database, chemical_type_dict,
                                                                            before_represent_chem)
        opt_temp = []
        for word in new_sent:
            operation = _get_operation(word.word)
            if operation:
                word._operation = operation[0]
                opt_temp.append(word)

        for prop, conditions in unit_dictionary["Condition"].items():
            for condition in conditions:
                matched_operation = word_nearest(condition, opt_temp, lambda t: True)

                if isinstance(matched_operation, Word) and matched_operation.distance_with_word(condition) < 20:
                    condition._operation = matched_operation._operation

        for word in new_sent:
            if word._operation is not None or isinstance(word, Value) and word.prop_type == 'Condition':
                operation = word._operation
                try:
                    assert operation_list[-1]['name'] == operation
                except (AssertionError, IndexError):
                    operation_list.append({'name': operation, 'condition': []})

                if isinstance(word, Value):
                    value_dict = {'Value': word.value, 'Unit': word.unit, 'Property': word.prop}
                    operation_list[-1]['condition'].append(value_dict)

        for prop, reactions in unit_dictionary["Reaction"].items():
            reaction = reactions[0]
            reaction_dictionary[reaction.prop] = {'Value': reaction.value, 'Unit': reaction.unit}

        for prop, conditions in unit_dictionary["General"].items():
            for condition in conditions:
                target, unit, value = str(condition.target), condition.unit, condition.value
                value_tuple = (value, unit)
                if target and target in general_dictionary:
                    general_dictionary[target].append(value_tuple)
                else:
                    general_dictionary[target] = [value_tuple]

    def get_condition_dictionary(operation_list):
        time = None
        temp = None
        time_op = None
        temp_op = None

        for operation_ in operation_list:
            op_name = operation_['name']
            if op_name in ['wash', 'purify', 'filter', 'dry', 'evaporate', 'diffuse']:
                break
            elif op_name not in [None, 'heat', 'wait']:
                continue

            for condition_ in operation_['condition']:
                property = condition_['Property']
                if property == 'Time':
                    if time is None:
                        time = {'Value': condition_['Value'], 'Unit': condition_['Unit']}
                        time_op = operation_['name']
                    elif not time_op and op_name == 'heat':
                        time = {'Value': condition_['Value'], 'Unit': condition_['Unit']}
                        time_op = operation_['name']
                elif property == 'Temperature':
                    if temp is None:
                        temp = {'Value': condition_['Value'], 'Unit': condition_['Unit']}
                        temp_op = operation_['name']
                    elif not temp_op and op_name == 'heat':
                        temp = {'Value': condition_['Value'], 'Unit': condition_['Unit']}
                        temp_op = operation_['name']
        return {'Time': time, 'Temperature': temp}

    condition_dictionary = get_condition_dictionary(operation_list)

    return condition_dictionary, general_dictionary, reaction_dictionary, operation_list


def _get_method(element):
    if isinstance(element, list):
        text = " ".join(element)
    elif isinstance(element, str):
        text = element
    else:
        raise TypeError()

    if regex.search(r"(?i)(?<=\b)(electr[oi](?!n)|cathode|anode|voltage)", element):
        return 'Electrochemical'
    elif regex.search(r"(?i)(?<=\b)micro[\s-]?wave", element):
        return 'Microwave'
    elif regex.search(r"(?i)(?<=\b)(grind|ground|ball|mill|mechan)", element):
        return 'Mechanochemical'
    elif regex.search(r"(?i)(?<=\b)((ultra)?sonic|sono\s?chemical)", element):
        return 'Sonochemical'
    elif regex.search(r"(?i)(?<=\b)(solvothermal|hydrothermal|autoclave|heat|teflon-?\s?line)", element):
        return 'Conventional solvothermal'
    else:
        return None


def _get_operation(element):
    if isinstance(element, list):
        text = " ".join(element)
    elif isinstance(element, str):
        text = element
    else:
        raise TypeError()

    operation_list = {'heat': 'heat', 'cool': 'cool', 'stir': 'stir', 'wash': 'wash', 'remov': 'remove',
                      'dehydrat': 'dehydrate', 'desicat': 'desicate', 'dissolv': 'dissolve', 'sonic': 'sonicate',
                      'ultrasonic': 'sonicate', 'diffus': 'diffuse', 'stor': 'store', 'wait': 'wait', 'left': 'wait',
                      'purif': 'purify', 'lins': 'linse', 'filter': 'filter', 'dri': 'dry', 'dry': 'dry',
                      'ground': 'ground', 'redissolv': 'dissolve', 'evaporat': 'evaporate', 'oven': 'heat',
                      'refrigerator': 'cool', 'crystalliz': 'crystallize', 'recrystalliz': 'crystallize',
                      'keep': 'wait', 'kept': 'wait', 'autoclave': 'heat', 'Teflon-lined': 'heat',
                      'teflon-lined': 'heat', 'solvothermal': 'heat', 'hydrothermal': 'heat', 'warm': 'heat',
                      'prepar': 'prepare'}

    operation_ = r"|".join(operation_list.keys())
    operation = regex.finditer(fr"(?i)(?<=^|\W)(?P<operation>{operation_})", text)

    result = []
    for oper in operation:
        oper_type = oper.group("operation")
        oper_type = oper_type.lower()
        result.append(operation_list[oper_type])
    return result


def replace_mof(mof_list):
    """ replace target and precursors from mof_list
    input : mof_list
    output : None
    """
    replace_words = ["except", "replace", "replaced", "substituted", "substitute", "instead", "similar", "same",
                     "identical", "change", "in place of"]

    name_symbol_mofs = []

    for i, now_mof in enumerate(mof_list):
        replace = False
        # 1. check replace words and check name or symbol of old mofs.
        name_symbol_old_mof = list(set(split_text(cleanup_text(now_mof._text))) & set(name_symbol_mofs))

        if any(replace_word in now_mof._text for replace_word in replace_words) and name_symbol_old_mof:
            # 2. save old_mof (first name_symbol_old_mof)
            for idx in range(i):
                if mof_list[idx].name == name_symbol_old_mof[0] or mof_list[idx].symbol == name_symbol_old_mof[0]:
                    old_mof = deepcopy(mof_list[idx])
                    replace = True

        # save name and symbol of old mofs
        if now_mof.name:
            name_symbol_mofs.append(now_mof.name)
        if now_mof.symbol:
            name_symbol_mofs.append(now_mof.symbol) 

        # 3. remove and append materials when replace is True
        if not replace:
            continue

        new_mof = deepcopy(old_mof)
        new_mof.name = now_mof.name
        new_mof.symbol = now_mof.symbol

        # 3.1 precursor

        remove_precursor = 0
        for precursor_now_mof in now_mof._precursor:
            # instead 뒤에 precursor가 now_mof's precursor로 뽑힐 경우 -> old_mof에서 제거해서 -> now_mof 에 저장
            if any(precursor_now_mof["name"] == precursor_old_mof["name"] for precursor_old_mof in old_mof._precursor):
                new_mof.remove_material(name=precursor_now_mof["name"])
                remove_precursor += 1
            else:
                new_mof.append_material(precursor_now_mof, astype="precursor")

        # instead 뒤에 percursor가 mof_._precursor로 안 뽑힐 경우
        if remove_precursor == 0:  # remove한게 하나도 없을 경우
            for precursor_old_mof in old_mof._precursor:
                if precursor_old_mof["name"] in split_text(cleanup_text(now_mof._text)):
                    new_mof.remove_material(name=precursor_old_mof["name"])

        # 3.2 etc (same as 3.1)
        remove_etc = 0
        for etc_now_mof in now_mof._etc:
            if any(etc_now_mof["name"] == etc_old_mof["name"] for etc_old_mof in old_mof._etc):
                new_mof.remove_material(name=etc_now_mof["name"])
                remove_etc += 1
            else:
                new_mof.append_material(etc_now_mof, astype="etc")

        if remove_etc == 0:
            for etc_old_mof in old_mof._etc:
                if etc_old_mof["name"] in split_text(cleanup_text(now_mof._text)):
                    new_mof.remove_material(name=etc_old_mof["name"])

        # 3.3 temperture and time
        if now_mof.temperature != (None, None):
            new_mof.temperature = now_mof.temperature
        if now_mof.time != (None, None):
            new_mof.time = now_mof.time

        mof_list[i] = new_mof
    
    return mof_list
