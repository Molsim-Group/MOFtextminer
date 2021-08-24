#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os

import regex as re
import json
from pathlib import Path

from collections import Counter
from functools import reduce
from itertools import tee, chain

import numpy as np

from bs4 import BeautifulSoup

#import spacy
from chemdataextractor import Document
from chemdataextractor.doc.text import Paragraph, Title, Sentence, Heading
from chemdataextractor.reader import PdfReader
#from chemdataextractor.nlp.tokenize import ChemWordTokenizer as tokenizer

from fuzzywuzzy import fuzz


from .utils import small_molecule_database, Get, Check, word_original
from .abbreviation import Abbreviation, make_abbreviation
from .unit_database import Unit_database


# In[8]:


#nlp = spacy.load("en_core_web_sm")
#nlp = spacy.load("en")

class DocumentTM():
    
    def database(self):
        database = {}
        list_dataset = ['chemname', 'chemhash', 'unit_database', 'ABB1', 'ABB2', 'ion1', 'ion2', 'sm1', 'sm2']
        for dic_name in list_dataset:
            exec("""database['{dic_name}'] = self._{dic_name}""".format(dic_name = dic_name))
        return database
    
    def set_dictionary(self, **database):
        abs_path = Path('./libs/property_extractor/database')
        
        if 'chemname' in database and 'chemhash' in database:
            self._chemname = database.get('chemname')
            self._chemhash = database.get('chemhash')
        else:
            self._chemname = {}
            self._chemhash = {}
        
        if 'sm1' in database and 'sm2' in database:
            self._sm1 = database.get('sm1')
            self._sm2 = database.get('sm2')
        else:
            with open(abs_path/"smallmolecule.json", 'r') as f:
                sm_database = json.load(f)
            self._sm1, self._sm2 = small_molecule_database(sm_database)
            
        if 'ion1' in database and 'ion2' in database:
            self._ion1 = database.get('ion1')
            self._ion2 = database.get('ion2')
        else:
            with open(abs_path/"ion_dict.json", 'r') as f:
                ion_database = json.load(f)
            self._ion1, self._ion2 = ion_database
        
        if 'ABB1' in database and 'ABB2' in database:
            self._ABB1 = database.get('ABB1')
            self._ABB2 = database.get('ABB2')
        else:
            with open(abs_path/"abb_list.json", 'r') as f:
                abb_list = json.load(f)
            self._ABB1, self._ABB2 = make_abbreviation(abb_list)
            
        self._ABB_re = self._change_to_re(self._ABB1.keys(), return_as_str=True)
        
        unit_database = database.get('unit_database', None)
        
        if unit_database:
            self._unit_database = unit_database
        else:
            self._unit_database = Unit_database(self._chemname, self._sm1, self._ion1)

    def __init__(self, file_name, **database):
        
        if isinstance(file_name, str):
            self._html_name = Path(file_name)
        elif isinstance (file_name, Path):
            self._html_name = file_name
        else:
            raise TypeError
        
        if self._html_name.suffix not in ['.pdf', '.html', '.xml']:
            raise TypeError 

        self.set_dictionary(**database)
        self._strange = Counter()
        self._chemical_counter = Counter()

        ELEMENTS = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K",
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

        MOLECULAR_NAMES = ["carbon dioxide", "carbon oxide", 'methan', 'methanol', 'ethanol']

        MOLECULAR = ["CO2", "CO", "NH3", "H2O2", "H2O", "CO3", "ClO3-", "KOH", "Cl2", "Cl", "Br2", "Br", "CH4", 'C2H5OH', 'HCl',
                     "NO2", "CaCO3", "NaCl", 'Ca\(OH\)2']

        ELEMENTS_AND_NAMES = MOLECULAR + MOLECULAR_NAMES + ELEMENTS + ELEMENT_NAMES + [en.capitalize() for en in ELEMENT_NAMES + MOLECULAR_NAMES]

        self.ELEMENTS_AND_NAMES = ELEMENTS_AND_NAMES
        self.MOLECULAR_NAMES = MOLECULAR_NAMES
        
        self.ELEMENT1 = ["H", "Li", "Na", "K", "Rb", "Cs", "Fr", "Be", "Mg", "Ca", "Sr", "Br", "Ra", "B","Al", "Ga", "C", "Si", "Ge", "Sn", "Pb",
                        "P","N", "As","Sb", "Bi",'O','S','Se','Te', 'F','Cl','Br','I','At','He','Ne','Ar','Kr',"Xe",'Rn']
        self.ELEMENT = ELEMENTS
        self.ELEMENT_NAMES = ELEMENT_NAMES

        self.MOLECULAR = MOLECULAR

        self.UNIT = ["days", "weeks", "hours", "minutes", "seconds", "Ω", "day", "week", "hour", "minute", "months", "month", "year", 
                "cycles", "years", 'bar', "Torr", 'vol', "mol", "℃", "℉", 'min', "Pa", 'cd', 'rad',"eV", "cal", 'PPM', 'ppm', "cc", "CC",
                     "m", 's', 'h', "K", "°C", "°F", "°", "g", "Hz", 'N', "V", "L", "l", "W", "%", "wt[.]", "wt", "A", "F", "Å" , "J",
                    'C', "M", 'atm']

        self.UNIT_PRX = ["m", "k", "G", "M", "c", "d", "n", "μ", 'µ', "T", "P", 'h', "p"]

        self.UNIT_mid = ["[(]STP[)]", "STP", "cat", '/', 'per']
        self.UNIT_sub = ["cat", "cats", "-cat", "[-]cat.", "[-]cat.", 'catalyst', "Cat", "CAT", r"\scat\s", "obs"]
        
        #global nlp
        #self.nlp = nlp
        
    def get_doc(self, parser = None):
        suffix = self._html_name.suffix
        if suffix == '.pdf':
            with open(str(self._html_name), 'rb') as f:
                doc = Document.from_file(f, readers = [PdfReader()])
                
        elif parser == 'cde_parser':
            with open(str(self._html_name), 'rb') as f:
                doc = Document.from_file(f)

        elif suffix == '.html':

            with open(str(self._html_name), encoding = 'UTF-8') as f:
                bs = BeautifulSoup(f, 'html.reader')

            for dl in bs.find_all('dl'):
                dl.decompose()
            for li in bs.find_all('li'):
                li.decompose()

            for k in bs.find_all('a'):

                if k.find('sup') or k.find('u'):
                    k.decompose()
                    continue

                text_k = k.get_text()
                if not len(text_k):
                    continue
                elif re.match(r"^S?(\d+[A-Za-z]?|[A-Za-z]\d*)$", text_k):
                    k.replace_with(" ")
                elif re.match(r"^[{\[(].+[)\]}]$", text_k):
                    k.replace_with(" ")

            # Bold font  : M -> M(bold)
            for bold in chain(bs.find_all('strong'), bs.find_all('b')):
                text_b = bold.get_text()
                bold.replace_with("{}(bold)".format(text_b))

            # Italic font  : M -> M(italic)
            """ for italic in chain(bs.find_all('em'), bs.find_all('i')):
                text_i = italic.get_text()
                italic.replace_with("{}(italic)".format(text_i))"""
            
            # Sub text : Msub -> M_{sub}
            for sub in bs.find_all('sub'):
                text_sub = sub.get_text()
                if not re.match(r"^[0-9x]+$", text_sub) and text_sub != 'x':
                    sub.replace_with(f"_[{text_sub}]")

            # Remove table 
            for table in bs.find_all('table'):
                table.decompose()

            temppath = 'tempfile.html'
            #with NamedTemporaryFile('w+', delete = False, encoding='UTF-8') as tempfile:
            with open(temppath, 'w', encoding='UTF-8') as tempfile:
                tempfile.write(str(bs))
                #temppath = tempfile.name

            with open(temppath, 'rb') as tempfile2:
                doc = Document.from_file(tempfile2)

            os.unlink(temppath)
            assert not os.path.exists(temppath)
        
        self.doc = doc
        self.Title = list(filter(lambda t : isinstance(t, Title), doc.elements))
        self.Para = list(filter(lambda t: isinstance(t, Paragraph), doc.elements))
        
    def set_special_unit(self, special_unit_dictionary):
        """set special unit in class <Unit_database>
        >>> input
        (dict) dictionary for special unit {unit_name : unit_regex}
        
        unit_name : (str) name of unit
        unit_regex : (real string) pattern of regular expression.
        
        * unit_regex must include group "NUM" (which is value of unit)
        ** Special unit would be replaced to -num('group_NUM')- -u******-
        *** Special unit would be defined as unit_name, dimension is [0,0,0,0,0,0,0]
            ex. unit_name = 'pH' -> -u001000- : {unit: 'pH', dimension: [0,0,0,0,0,0,0]}
        
        """
        
        for unit_name, unit_regex in special_unit_dictionary.items():
            self._unit_database.add_special_unit(unit_name, unit_regex)
        
        
    def _change_to_re(self, chemical_list, original_list=[], return_as_str = False):
        """---input--------------
        chemical_list : (list) list of chemical
        original_list : (list) list that origial change_to_re list
        return_as_str : (bool) if True, return (str). Else, return list
        
        """
        new_list = []
        if isinstance(chemical_list, str):
            chemical_list = [chemical_list]
            
        for chemical in chemical_list:
            if not chemical:
                print (chemical_revised)
                continue
            elif not isinstance(chemical, str):
                chemical = str(chemical)
                
            chemical_revised = re.sub(pattern=r"(?:\[|\]|\(|\)|\.|\,|\-|\*|\?|\{|\}|\$|\^|[|]|\+|\\)", 
                                      string = chemical, repl=lambda t: r"\{}".format(t.group()))
        
            if chemical_revised in original_list or chemical_revised in new_list:
                continue
            elif re.findall(pattern=chemical_revised, string=chemical):
                    new_list.append(chemical_revised)
            else:
                print ("re compile error", chemical)
        
        #should be revised
        new_list += original_list
        new_list = sorted(new_list, reverse=True)
        
        if return_as_str:
            return r"|".join(new_list)
        else:
            return new_list
        
    def tokenize_CDE(self, text):
        
        temp_split = text.split()
        new_split = []
        activation = False
        for word in temp_split:
            if re.match(r"^-(num|strange)\(", word):
                if re.match(r".+\)-$", word):
                    pass
                else:
                    activation = True
                new_split.append(word)
            elif activation:
                if re.match(r".+\)-$", word):
                    activation = False
                new_split[-1] += " "+word
            else:
                new_split.append(word)
            
        #new_split = [word for word in re.split(r"((?:-(?:num|strange)\(.+\)-)|\s)", text) if word and not re.match(r"^\s+$", word)]
        
        return new_split     
            
    def find_strange(self, text = None, cut_off = 3):
        strange_counter = Counter()
        self._strange = set()
        
        if not text:
            Para = self.Para
        else:
            Para = [text]
            
        text_list = map(lambda p: self.paragraph_preprocessing(p) , Para)
        
        for text in text_list:
            split = self.tokenize_CDE(text)
            
            word_strange = filter(lambda t: not re.search(r"^(?:(?:[±×+.-]|\d)+|\d+(st|rd|nd|th)|-.+?-|.+_\[.+\].*|([Ff]igure|[Tt]able|[Ff]ig|ESI)\S+|[A-Z]?[a-z]+([-/][A-Z]?[a-z]+)+|[A-Z]?[a-z]+)$", t) and len(t) > 1, split)
            a1, a2 = tee(word_strange)
            strange_counter.update(chain.from_iterable(map(lambda word: [z.group("name")for z in re.finditer(r"(?P<name>([A-Za-z0-9])*[A-Za-z](?(1)|[A-Za-z0-9])*)", word)], a2)))
            self._strange = self._strange.union(a1)

            
        for word in iter(self._strange.copy()):
            word_analysis = [z.group("name")for z in re.finditer(r"(?P<name>([A-Za-z0-9])*[A-Za-z](?(1)|[A-Za-z0-9])*)", word)]
            if not word_analysis:
                self._strange.remove(word)
            elif reduce(lambda a, t: a + strange_counter[t], word_analysis, 0)/len(word_analysis) < cut_off:
                self._strange.remove(word)
            
    @staticmethod
    def hash_code(label):
        num = np.random.randint(1000000)
        return "-"+label[0]+str(num).zfill(6)+"-"
    
    @staticmethod
    def identify_tag(label):
        
        if not label:
            return None

        if re.match(r"^-num([(].+[)])?-$", label): #임시사용
            return "number"
        elif re.match(r"^-pH(.+)-$", label):
            return "pH"
        elif re.match(r"^-strange\(.+\)-$", label):
            return 'strange'
        
        tag = re.match(r"^[-](?P<tag>c|u|n|a|e|i|s)\d+[-]$", label)
        
        if not tag:
            return None
        
        tag_v = tag.group('tag')
        tag_dict = {'c':'chemical', 'n':'number', 'u':'unit', 'e':'element', 's':'smallmolecule', 'i':'ion'}
        
        return tag_dict[tag_v]
    
    def chemical_hash(self, chemical_name, hashcode = None, strange_off = False):
        chemical_name = re.sub(r"\(bold\)", "", chemical_name)
        
        if chemical_name in ['estimated', 'trend', 'discovery', 'bore', 'and', 'end', ',', 'Table',
                        'retain', 'backbone', 'scene', 'lamps', 'zip', 'ribbon', 'school', 'join', 'cyan', 'theme', 'plugin',
                        'ramp', 'doc', 'constant', 'constance', 'visible-light', 'measuring', 'estimate']:
            return chemical_name
        elif chemical_name in self._chemname: # Already existed
            return self._chemname[chemical_name]
        elif chemical_name in self._sm1:
            return self._sm1[chemical_name]
        elif chemical_name in self._ion1:
            return self._ion1[chemical_name]
        elif chemical_name in self.ELEMENT:
            return "-e{}-".format(str(self.ELEMENT.index(chemical_name)).zfill(6))
        elif not strange_off and chemical_name in self._strange:
            return "-strange({})-".format(chemical_name)
        
        while not hashcode or hashcode in self._chemhash: # Get hashcode
            hashcode = self.hash_code('chemical')
        
        self._chemname[chemical_name] = hashcode
        self._chemhash[hashcode] = chemical_name

        self.ELEMENTS_AND_NAMES=self._change_to_re([chemical_name],self.ELEMENTS_AND_NAMES)
        return hashcode
    
    def ion_hash(self, ion_name, hashcode = None):
        
        if ion_name in self._ion1: # Already existed
            return self._ion1[ion_name]
        
        while not hashcode or hashcode in self._ion1: # Get hashcode
            hashcode = self.hash_code('ion')
            
        self._ion1[ion_name] = hashcode
        self._ion2[hashcode] = ion_name
        
        self.ELEMENTS_AND_NAMES=self._change_to_re([ion_name],self.ELEMENTS_AND_NAMES)
        
        
        return hashcode
    
    def sm_hash(self, sm_name, hashcode = None):
        
        if sm_name in self._sm1: # Already existed
            return self._sm1[sm_name]
        
        while not hashcode or hashcode in self._sm1: # Get hashcode
            hashcode = self.hash_code('sm')
            
        self._sm1[sm_name] = hashcode
        self._sm2[hashcode] = sm_name
        
        self.ELEMENTS_AND_NAMES=self._change_to_re([sm_name],self.ELEMENTS_AND_NAMES)
        
        return hashcode 
    
    def ABB_hash(self, ABB_tuple, hashcode = None):
        
        ABB_name = " ".join(ABB_tuple[0])
        ABB_name2 = " ".join(ABB_tuple[1])
        ABB_type = ABB_tuple[2]
        
        if len(ABB_tuple[0]) == 1:
            if len(ABB_tuple[1]) == 1 and len(ABB_name) > len(ABB_name2):
                ABB_def = ABB_name
                ABB = ABB_name2 
                
            else:
                ABB_def = ABB_name2
                ABB = ABB_name
            
        elif len(ABB_tuple[1]) == 1:
            ABB_def = ABB_name
            ABB = ABB_name2
            
        else:
            #print ("ABB error : ", ABB_name, ABB_name2)
            return None
        
        # validation_check
        if len(re.findall(r"\(|\{|\[", ABB_def)) != len(re.findall(r"\)|\]|\}", ABB_def)):
            return None
    
        #ABB_front_char = reduce(lambda x, y: x + "".join(re.findall(r"^\S|[A-Z]", y)), re.split(r",|\s|-", ABB_def), "")
        ABB_front_char = reduce(lambda x, y: x + "".join(re.findall(r"^\S|[^a-z]", y)), re.split(r",|\s|-", ABB_def), "")
        if ABB[-1] == 's' and ABB_def[-1] == 's':
            ABB_front_char += 's'
        ABB_char = re.sub(r"(,|\s|[-])", "", ABB)
        ratio = fuzz.ratio(ABB_char.lower(), ABB_front_char.lower())
        
        #print (ABB_def, re.findall(r"(?<=^|\s|-)({})(?=\s|-|$)".format(r"|".join(self.ELEMENTS_AND_NAMES)), ABB_def))
        
        if ratio < 70 and not ABB_type:
            return None
        
        if re.findall(r"(?i)reaction", ABB_def):
            ABB_type = None
        elif ABB_type:
            if not Paragraph(ABB_def.replace("-", " - ")).cems:
                ABB_type = None
        elif re.findall(r"(?<=^|\s|-)({})(?=\s|-|$)".format(r"|".join(self.ELEMENTS_AND_NAMES)), ABB_def):
            ABB_type = 'CM'
        
        
        if ABB in self._ABB2:
            new_abb = self._ABB2[ABB]
            new_abb.update(ABB_def, ABB_type)
            self._ABB1[ABB_def] = ABB
            self._ABB_re = self._change_to_re(self._ABB1.keys(), return_as_str=True)
            
            return None
            
        else:
            new_abb = Abbreviation(ABB, ABB_def, ABB_type)
            if not len(new_abb):
                return None
            self._ABB2[ABB] = new_abb
            self._ABB1[ABB] = ABB
            self._ABB1[ABB_def] = ABB
            self._ABB_re = self._change_to_re(self._ABB1.keys(), return_as_str=True)
            
            if ABB_type:
                chem_hash = self.chemical_hash(ABB, strange_off = True)
                self.chemical_hash(ABB_def, hashcode = chem_hash, strange_off = True)
            
            return (new_abb, ABB_def, ABB_type)
    
    def get_name(self, name):
        tag = self.identify_tag(name)
        if tag == 'chemical':
            return self._chemhash[name]
        elif tag == 'abbreviation':
            return None
        elif tag == 'unit':
            return self._unit_database[name]['unit']

        elif tag == 'number':
            num =  re.match(r"^-num[(](?P<num>.+)[)]-$", name)
            if num:
                return num.group("num")
            
            return "-num-"  #should be revised
        
        elif tag == 'element':
            index = re.findall(r"-e(\d+)-", name)[0]
            return self.ELEMENT[int(index)]
        
        elif tag == 'smallmolecule':
            return self._sm2[name]
        elif tag == 'ion':
            return self._ion2[name]
        elif tag == 'strange':
            strange = re.match(r"^-strange[(](?P<strange>.+)[)]-$", name)
            if strange:
                return strange.group("strange")
        else:
            return name
        
    def get_original_token(self, iterable):
        return [self.get_name(word) for word in iterable]
        
    def num_tree(self):
        def Tree_maker(iterator, Tree ={None:False}):
            if not isinstance(iterator, str):
                for node in iterator:
                    Tree = Tree_maker(node, Tree)
                return Tree

            iterator = [None] + iterator.split()
            iterator.reverse() 

            childtree = Tree
            for word in iterator:
                if not word:
                    childtree[word] = True

                elif word in childtree:
                    childtree = childtree[word]
                else:
                    childtree[word] = {None:False}
                    childtree = childtree[word]
            return Tree

        wordlist = ["about", "ca .", "ca.", "less than", "more than", "approximately", "around", "roughly", "up to", "nearly", "~", "over",
                   "average", "equal to", "only", "great than", "close to", "correspond to", "twice", "only ~", "maximum at", "maximum",
                   "estimate to be", "below", "in range of", "almost", "only about", "at around" "maximum of", "minimum of", 'close to', 'estimate to be',
                   'decrease to', 'increase to', 'calculate to be', 'high than that of', 'small than', 'as high as', 'slightly', 'circa','measure to be',
                   'large than', 'only', 'decrease by', 'observe at', 'decrease from', 'range between', 'relative to', 'increase from', 'increase to',
                   'as determine by', 'large than that of', 'up to ~', 'vary between', 'amount to', 'on average', 'value of', 'equal', 'equal at',
                   'at least', 'measure at', 'approximate', 'equivalent to', 'increase to', 'larger than', 'constant at', 'center at', 'determine to be',
                   'slightly', 'as low as', 'slightly high', 'approximate', 'high than that of', 'in range from', 'vary from', 'low than', 'above',
                   '>', '<', 'no more than', 'no less than', 'no greater than', 'not increase to', 'not decrease to', 'no large than', 'greater than',]

        return Tree_maker(wordlist)
    

    #############################################################################################################
    def paragraph_preprocessing(self, Para):
        if isinstance(Para, str):
            Para = Paragraph(Para)  
        elif not isinstance(Para, Paragraph):
            return None

        def remove_dot(sentence):
            text = sentence.text
            if text[-1] in ".!?":
                text = text[:-1]
            return text

        text = " -end- ".join(map(lambda t: remove_dot(t), Para.sentences))
        
        text = text.replace("\n", " ")
        
        text = re.sub(r"i\.e\.(?=,|\s)", "that is", text)
        text = re.sub(r"e\.g\.(?=,|\s)", "for example", text)
        text = re.sub(r"al\.", "al. and", text)
        text = re.sub(r"vs\s?\.", 'vs', text)
        
        # Change unicode;
        text = re.sub(r"[\u2000-\u2005\u2007\u2008]|\xa0", " ", text)
        text = re.sub(r"[\u2006\u2009]", "", text)
        text = re.sub(r"[\u2010-\u2015]|\u2212", "-", text)
        text = re.sub(r"≈|∼","~", text)
        text = re.sub(r"\u2032|\u201B|\u2019", "'", text)
        text = re.sub(r"\u2215", "/", text)
        text = re.sub(r"\u201A", ',', text)
        text = re.sub(r"\u201C|\u201D|\u2033", '"', text)
        #text = re.sub(r"°C", "℃", text)
        #text = re.sub(r"°F", "℉", text)
        
        text = re.sub(pattern=r"\s?\b[.][,-]+\s?", string = text, repl=" -end- ")
        text = re.sub(r"\[\s(\,\s)*\]", repl='', string=text)
        text = re.sub(r"\b(([A-Za-z]{2,})[,.]\d+[A-Za-z]?)\b", string=text, repl=lambda t: t.group(2)+" -end- ")
        text = re.sub(r"(?<=\s)[.,-](?=\s)", " ", string=text)

        # Seperate equality, inequality sign 
        text = re.sub(r"(\s?[\u221D-\u22C3\u22CD-\u22FF:;=><~]\s?)", string = text, repl=lambda t:" {} ".format(t.group(1)))
        
        #text = re.sub(r"(\s?[><]\s?)(?=[±×+-]?\s?\d+)", lambda t: " = "+t.group(1)+" ", text)

        # Figure 3a -> Figure
        index = r"(\d+[a-z]?|[a-z]\d+?)\b(?:\s?\([A-Za-z1-9]\))?"

        index = r"(?:\d+|[A-Za-z]){1,2}\b(?:\s?[(][A-Za-z1-9][)]\B)?"
        figure_table = r"(\b([Ff]igure|[Tt]able|[Gg]raph|[Ss]cheme|[Ff]ig)(s|es)?(?:(?:[,]|and|[.]|\s|[-]|\B)+(?:S?\d+|[A-Za-z]){1,2}\b(?:\s?[(][A-Za-z1-9][)]\B)?)+?)"
        text = re.sub(figure_table, string = text, repl=lambda t: t.group(2))
        text = re.sub(r"(([(])?(Fig|Table|Figure|ESI)\s?†(?(2)[)]))", string=text, repl="")
        
        # 3p1/2 -> porbital
        text = re.sub(r"([1-9]([spdfg])\s?[13579]\s?[/]\s?2)", string = text, repl=lambda t:"{}'orbital".format(t.group(2)))

        # type i curve : -type-
        roman_num = r"\b(?:iv|ix|(?:v|x)i{0,3}|i{1,3})\b"
        types = r"((?i)type(?:\s-\s|-|\s)?([(])?{roman_num}(?(2)[)])(?:(?:\s-\s|-|\s)?curve)?)".format(roman_num=roman_num)
        text = re.sub(types, string=text, repl=" -type-")

        # Lee et al. -> -paper-
        text = re.sub(pattern=r"(\s?\b\S+\set\.?\sal(\.)?(?(2)\B|\b)\s?)", string = text, repl=" -paper- ")
        text = re.sub(r"((https?://)?\S+[.](com|net|org|edu)(/\S+)?)(?=\s|$)"," -email- ",text)
        
        text = re.sub(r"(?P<atom>1H|2H|6Li|11B|13C|15N|19F|29Si|31P|195Pt)\s?\S*(nmr|NMR)", lambda t: "{}-NMR".format(t.group("atom")), text)
        
        # .,:;
        #text = re.sub(r'(?P<name>(?<=\S+)[.,:;"](?=\S+))|[.,:;"]', lambda t: t.group() if t.group("name") else " {} ".format(t.group()),text)
        text = re.sub(r'(?P<name>(?<=\S+)[.,:;"](?=\S+))|(?P<OL>(?<=(^|\s|\(|\))[A-Za-z])[."](?=\s|\(|\)|$))|[.,:;"]', lambda t: t.group() if t.group("name") or t.group("OL") else " {} ".format(t.group()),text)
        
        # pH 5 -> -pH5.0-
        #text = re.sub(r"pH\s?(?P<num>-?\d+[.]?\d*)", lambda t: "-pH({})-".format(float(t.group("num"))), text)
        
        # C/10 -> 0.1 C
        text = re.sub(r"(?<=^|\s)C/(?P<num>\d+)(?=\s|$)", lambda t: "%.2f C"%(1/int(t.group("num"))), text)
    
        
        # First Letter lower
        text = re.sub(r"(?<=(^|-end-)\s*)(?P<word>[A-Z][a-z]+)(?=\s|$)", lambda t: t.group('word').lower(), text)
        
        # Remove possessive
        text = re.sub(r"(?<=\S+)'s(?=\s|$)", " ", text)
        
        len_text = len(text)
        Queue = []
        small_queue = []
        activation = True
        first_activation = False
        new_sent = ""
        poly_activation = None

        for i, char in enumerate(text):
            if char in '([{':
                small_queue.append(i)
                
                if Get(text, (i-4, i)) == 'poly': # Revise to poly!
                    poly_activation = i
                    
                elif activation:
                    first_activation = True
                    activation = False
                
                
            elif char == " " and not poly_activation:
                #print ('2', char, poly_activation, "3", text[:i], "4")
                activation = True
                first_activation = False
                
                if small_queue:
                    Queue += small_queue
                    small_queue.clear()

            elif char in ')]}':
                if small_queue:
                    pop_index = small_queue.pop()
                    
                    if poly_activation and not (poly_activation - pop_index):
                        #print ("poly_inactivated")
                        poly_activation = None
                            
                    elif not small_queue and first_activation:
                        if Get(text, i+1, " ") == " ":
                            Queue.append(pop_index)
                            Queue.append(i)

                        first_activation = False        
                else: # Already removed
                    Queue.append(i)
            else:
                activation = False

        Queue += small_queue

        for i, char in enumerate(text):
            if i in Queue:
                new_sent += " {} ".format(char)
            else:
                new_sent += char

        # Remove 'a' and 'the'
        new_sent = re.sub(r"(?<=^|\s)(?:a|the|an)(?=\s|$)", string = new_sent, repl=" ").strip()
        
        return new_sent
    
    def find_unit_from_text(self, text):
        unit_sub = self.UNIT_sub + self.ELEMENTS_AND_NAMES
        unit, text = self._unit_database.find_unit(text, unit_sub)
        
        return text, unit
    
    def find_abbreviation_from_text(self, text):
        def _ABB_sub(group):
            ABB_def = group.group("ABB")
            
            change_word = self._ABB1.get(ABB_def)
            if change_word:
                return change_word
            
            change_word = self._ABB1.get(ABB_def.lower())
            if change_word:
                return change_word
            
            return ABB_def
            
            
        #print (re.sub(fr"(?<=^|\s)(?P<ABB>{self._ABB_re})(?=\s|$)", _ABB_sub, text))
        text = re.sub(fr"(?<=^|\s)(?P<ABB>{self._ABB_re})(?=\s|$)", _ABB_sub, text)
        
        sentence = Paragraph(text)
        
        # Find (ABB)
        ABB = sentence.abbreviation_definitions
        new_abb_list = []
        for abb in ABB:
            abb_tuple = self.ABB_hash(abb)
            if abb_tuple:
                new_abb_list.append(abb_tuple)
                
        return text, new_abb_list
        

    ############################################################################################################
    def tokenize_paragraph(self, sentence, cut_off=True, lemma = True, Strange = False):
        if isinstance(sentence, str):
            sentence = Paragraph(sentence)
        
        text = self.paragraph_preprocessing(sentence)
        
        #print ('-2', text)
        
        if not text: 
            return (None, None)
            
        if cut_off:
            if len(sentence.sentences) < 3:
                return (None, None)
            if len(re.compile(r"[äüö]").findall(text)) > 3:
                return (None, None)
        
        # find ABB
        text, new_abb_list = self.find_abbreviation_from_text(text)
                
        # find unit
        text, unit = self.find_unit_from_text(text)

        #print('-1', text)
        split = self.tokenize_CDE(text)
        
        #print ('0', split)

        
        for i, word in enumerate(split):
            tag = self.identify_tag(word)
            
            #print (tag, word)
            if tag:
                continue
                
            elif word in self.ELEMENT:
                if i != 0 and split[i-1] != "-end-":
                    split[i] = "-e{}-".format(str(self.ELEMENT.index(word)).zfill(6))
                continue
                
            elif word in self._sm1:
                word = self._sm1[word]
                split[i] = word
                continue
                
            elif word in self._ion1:
                word = self._ion1[word]
                split[i] = word
                continue
            
            elif word in self._ABB1:
                abb_name = self._ABB1[word]
                abb = self._ABB2[abb_name]
                if abb.ABB_type == "CM":
                    word = self.chemical_hash(abb_name, strange_off=True)
                    split[i] = word
                continue
                
                    
            elif word in self._chemname:
                word = self._chemname[word]
                split[i] = word
                continue
            
            #print (word)
            
            # should be Revised
            num_string = "first second third forth fifth ninth twelfth one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen ninteen twenty thirty fourty fifty sixty seventy ninety hundred".split()
            num_string = r"|".join(num_string)
            #num_match = re.match(fr"^(?P<num>([(±×+-]?\s?\d+[.]?\d*[)]?\s?|({num_string})\s?)+(st|nd|rd|th)?)\s?$", word)
            num_match = re.match(fr"^(?P<num>[±×+-]?\s?(\d+\.\d+|\d+/\d+|\d+|{num_string})(st|nd|rd|th)?(?:\(\d+\))?)$", word)
            
            if num_match:
                split[i] = "-num({})-".format(num_match.group("num"))
                continue
            
            
            chemical = Paragraph(word.replace("-", " - ")).cems
            #print (word, chemical)
            
            if not chemical:
                if word in self._ABB2:
                    continue
                    
                elif Strange and word in self._strange:
                    word = re.sub(r"\(bold\)","", word)
                    split[i] = "-strange({})-".format(word)
                    self.ELEMENTS_AND_NAMES = self._change_to_re([word],self.ELEMENTS_AND_NAMES)

                continue
            
            if re.findall(r"^(?:(?:{element})\d*)+[+-]$".format(element=r"|".join(self.ELEMENT)), word):
                hashcode = self.ion_hash(word)
                
            elif re.findall(r"^(?:(?:{element})|{molecular})\d*(?:\b|\s?[(][\u2160-\u217FIXVixv]{iter}[)]\B)$".format(element=r"|".join(self.ELEMENT), molecular = r"|".join(self.MOLECULAR), iter=r"{1,5}"), word):
                hashcode = self.sm_hash(word)
                
            elif re.findall(r"^(?i)({})(s|es)?$".format(r"|".join(self.ELEMENT_NAMES)) , word):
                hashcode = self.sm_hash(word)
                
            else:
                hashcode = self.chemical_hash(word, strange_off=True)
            
            word = hashcode
            split[i] = word
            continue  
        
        for abb, abb_def, abb_type in new_abb_list:
            if abb_type:
                continue
                
            #print (abb, abb_def)
            new_abb_type = re.findall(r"(?<=^|\s|-)({})(?=\s|-|$)".format(r"|".join(self.ELEMENTS_AND_NAMES)), abb_def)
            if not new_abb_type:
                continue
            elif re.findall(r"(?i)reaction", abb_def):
                continue
            
            abb.change_ABB_type(abb_def, 'CM')
            abb_name = abb.ABB_name
            hash_code = self._chemname.get(abb_def, None)
            abb_cem = self.chemical_hash(abb_name, strange_off=True, hashcode = hash_code)
            for i, word in enumerate(split):
                if word == abb_name:
                    split[i] = abb_cem
                       
        
        #print ("1", split)
        
        if lemma:
            raise SyntaxError("Can not use function 'lemma' for these version")
            split_copy = split[:]
            lemma_activation = map(lambda token : bool(self.identify_tag(token)), split)
            lemma_sentence = " /// ".join(split)
            lemma_split = map(lambda token : token.string.strip() if self.identify_tag(token.string.strip()) else token.lemma_, nlp(lemma_sentence))
            #lemma_split = [token.lemma_ for token in nlp(lemma_sentence)]
            
            split = []
            activation = True
            for word in lemma_split:
                if word == "///":
                    activation = True
                elif activation:
                    split.append(word)
                    activation = False
                else:
                    split[-1]+= word
            for i, word in enumerate(split_copy):
                if next(lemma_activation):
                    split[i] = word
        
        # Distinguish 1 chemical 
        
        #print ("check2", split)
        
        activation = False
        bar_activation = False
        slash_activation = False
        a_activation = False
        
        
        chemical_prefix = ['metal', "poly", 'pure', 'bare', 'pristine', 'exfoliate', 'polymeric', 'coated', 'activated', 'modified',
                              'stablized', '1D', '2D', '3D', '1-D', '2-D', '3-D']
        
        for i, word in enumerate(split):
                        
            if not word:
                continue
                
            tag = self.identify_tag(word)
            
            #chemical_TF = tag in ['chemical', 'element', 'smallmolecule', 'ion', 'strange'] or word in chemical_prefix
            chemical_TF = tag in ['chemical', 'element', 'smallmolecule', 'ion', 'strange']
            
            #print ("activation:{}, bar:{}, slash:{}, chemical:{}".format(activation, bar_activation, slash_activation, str(chemical_TF)))
            #print (word, split, "\n\n")
             
            if not activation:
                
                if chemical_TF:
                    activation = True
                    if not i:
                        continue
                    before_word = self.get_name(Get(split, i-1))

                    # Prefix of chemical such as 'modified'
                    
                    #print (isinstance(before_word, str))
                    if isinstance(before_word, str):
                        if before_word in ['metal', "poly", 'pure', 'bare', 'pristine', 'exfoliate'] or re.match(r"\S+(ed|ic)$", before_word):
                            chem = "{} {}".format(before_word, self.get_name(split[i]))
                            split[i] = chem
                            split[i-1] = None
                    

                elif re.compile("^[\u2160-\u217FIXVixv]{1,5}$").match(word):

                    if split[i-1] == "(" and self.identify_tag(split[i-2]) in ['chemical', 'element', 'smallmolcule', 'ion'] and split[i+1] == ")":

                        chem = self.get_name(split[i-2]) + " " + split[i-1] + self.get_name(split[i]) + split[i+1]

                        """print ("checking for IVIVIVIVIVIVI") #Remove
                        print (chem)"""

                        split[i-2], split[i-1], split[i] = None, None, None

                        hashcode = self.chemical_hash(chem)

                        split[i+1] = hashcode

                elif tag == 'number':
                    num_tree = self.num_tree()
                    childtree = num_tree
                    
                    if self.identify_tag(Get(split, i-1)) == 'number':
                        split [i] = "-num({} {})-".format(self.get_name(split[i-1]), self.get_name(split[i])) 
                        split[i-1] = None
                        continue
                    
                    if split[i-1] in ['to', '-'] and self.identify_tag(split[i-2]) == 'number':
                        split[i] = "-num({})-".format(self.get_name(split[i-2]) + " " + split[i-1] + " "+self.get_name(split[i]))
                        split[i-2], split[i-1] = None, None
                        
                        if Get(split, i+1) in ['and', 'or'] and self.identify_tag(Get(split,i+2)) == 'number' and not Get(split, i+3) in ['to', '-']:
                            split[i] = "-num({})-".format(self.get_name(Get(split, i)) + " " + Get(split, i+1) + " "+self.get_name(Get(split, i+2)))
                            split[i+1], split[i+2] = None, None
                            
                        continue
                        
                    iter_num = 1
                    word = split[i-iter_num]
                    
                    while word in childtree:
                        childtree = childtree[word]
                        iter_num += 1
                        word = split[i-iter_num]
                    
                    if childtree[None]:
                        revised_word = ""
                        for iternum in range(1, iter_num):
                            revised_word = split[i-iternum] + " " + revised_word
                            split[i-iternum] = None
                        revised_word += self.get_name(split[i])
                        split[i] = "-num({})-".format(revised_word)
                else:
                    continue
                    
            # Activation_case        
                
            elif not chemical_TF:
                # Suffix of chemical
                suffix_list = r"|".join(['acid', "'", 'oxide', 'dioxide', 'monoxide', 'trioxide', 'powder', 'crystal', 'crystalline', 'particle', 'hollow', 
                               'sphere','film', 'web', 'sheet', 'flower', 'fiber', 'atom', 'pore', 'composite', 'N[A-Z]s', 'metal'])
                
                sub_chemical_suffix_list = ['in', 'on', 'based', 'layered', 'coated', 'with', 'decorated', 'activated', 
                                            'deposited', 'combined', 'supported']
                
                if re.match(fr"(?i)(ion|anion|cation)(s|es|ies)?", word):
                    if self.identify_tag(split[i-1]) == 'ion':
                        split[i] = split[i-1]
                        split[i-1] = None
                    else:
                        ion_name = self.get_name(split[i-1])+" " + self.get_name(split[i])
                        HashIon = self.ion_hash(ion_name)
                        split[i] = HashIon
                        split[i-1] = None
                        
                elif re.match(fr"({suffix_list})(s|es|ies)?", word) or re.match(r"^(nano|micro|macro|meso)\S+", word):
                    split[i] = self.get_name(split[i-1])+" " + self.get_name(split[i])
                    split[i-1] = None
                    
                elif word in sub_chemical_suffix_list:
                    next_word = Get(split, i+1)
                
                    if next_word in sub_chemical_suffix_list or self.identify_tag(next_word) in ['chemical', 'element', 'smallmolecule', 'ion', 'strange']:
                        split[i] = self.get_name(split[i-1])+" " + self.get_name(split[i])
                        split[i-1] = None
                    else:
                        activation = False
                        
                        chemname = self.get_name(split[i-1])
                        hashcode = self.chemical_hash(chemname)
                        split[i-1] = hashcode     
                else: #split[i] = self.get_name(split[i-1])+" " + self.get_name(split[i])
                    
                    activation = False
                    chemname = self.get_name(split[i-1])
                    hashcode = self.chemical_hash(chemname)
                    split[i-1] = hashcode
                    
            else:
                split[i] = self.get_name(split[i-1])+ " " + self.get_name(split[i])
                split[i-1] = None
                #print (new_split[i])
        
        if activation and split[-1]: #Remove last term
            #if not bar_activation and not slash_activation:
            chemname = self.get_name(split[-1])
            hashcode = self.chemical_hash(chemname)
            split[-1] = hashcode
            activation = False
            
        new_split = list(filter(lambda t : t and t != " ", split))
        
        
        #print ("3", new_split)
        
        for i, word in enumerate(new_split):
            tag = self.identify_tag(word)
            if tag == 'chemical' or tag == 'strange':
                name = self.get_name(word)
                
                self.sent = new_split # Should be revised
                
                if " " in name:
                    chem_analyize = self.find_chemical_form(name, tag).get('chemical', None)
                else:
                    chem_analyize = self.is_vaild_chemical(name, tag)
                #print (name, chem_analyize)
                if chem_analyize:
                    self._chemical_counter.update([name])
                else:
                    new_split[i] = name
        
        #print ("4", new_split)
        
        return new_split, unit
    
    
    def tokenize_test(self, sentence, cut_off=True, lemma = True, Strange = False):
        if isinstance(sentence, str):
            sentence = Paragraph(sentence)
        
        text = self.paragraph_preprocessing(sentence)
        
        #print ('-2', text)
        
        if not text: 
            return (None, None)
            
        if cut_off:
            if len(sentence.sentences) < 3:
                return (None, None)
            if len(re.compile(r"[äüö]").findall(text)) > 3:
                return (None, None)
        
        def _ABB_sub(group):
            ABB_def = group.group("ABB")
            
            change_word = self._ABB1.get(ABB_def)
            if change_word:
                return change_word
            
            change_word = self._ABB1.get(ABB_def.lower())
            if change_word:
                return change_word
            
            return ABB_def
            
            
        #print (re.sub(fr"(?<=^|\s)(?P<ABB>{self._ABB_re})(?=\s|$)", _ABB_sub, text))
        text = re.sub(fr"(?<=^|\s)(?P<ABB>{self._ABB_re})(?=\s|$)", _ABB_sub, text)
        
        sentence = Paragraph(text)

        # Find (ABB)
        ABB = sentence.abbreviation_definitions
        new_abb_list = []
        for abb in ABB:
            abb_tuple = self.ABB_hash(abb)
            if abb_tuple:
                new_abb_list.append(abb_tuple)
        
        # initial -> 1st -> 1
        text = re.sub(r"(?i)initial", "1st", text)
        # room temperature -> 295 K
        text = re.sub(r"(?i)room temperature", " -num(room temperature)- K ", text)
        
        # find unit
        unit_sub = self.UNIT_sub + self.ELEMENTS_AND_NAMES
        unit, text = self._unit_database.find_unit(text, unit_sub)

        #print('-1', text)
        
        for chemical in sentence.cems:
            hash_c = self.chemical_hash(str(chemical), strange_off=True)
        
        re_cems = r"(?<=^|\s)(?P<cem>{})(?=\s|$)".format(r"|".join(self.ELEMENTS_AND_NAMES))
        
        text = re.sub(re_cems, lambda t: self.chemical_hash(t.group('cem')), text)

        #print (text)
        split = self.tokenize_CDE(text)
        
        num_string = "first second third forth fifth ninth twelfth one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen ninteen twenty thirty fourty fifty sixty seventy ninety hundred".split()
        num_string = r"|".join(num_string)
        
        for i, word in enumerate(split):
            
            num_match = re.match(fr"^(?P<num>[±×+-]?\s?(\d+\.\d+|\d+/\d+|\d+|{num_string})(st|nd|rd|th)?(?:\(\d+\))?)$", word)
            
            if num_match:
                split[i] = "-num({})-".format(num_match.group("num"))
            elif re.match(r'-num(.+)-', word):
                pass
            else:
                continue
                
            num_tree = self.num_tree()
            childtree = num_tree

            if self.identify_tag(Get(split, i-1)) == 'number':
                split [i] = "-num({} {})-".format(self.get_name(split[i-1]), self.get_name(split[i])) 
                split[i-1] = None
                continue

            if split[i-1] in ['to', '-'] and self.identify_tag(split[i-2]) == 'number':
                split[i] = "-num({})-".format(self.get_name(split[i-2]) + " " + split[i-1] + " "+self.get_name(split[i]))
                split[i-2], split[i-1] = None, None

                if Get(split, i+1) in ['and', 'or'] and self.identify_tag(Get(split,i+2)) == 'number' and not Get(split, i+3) in ['to', '-']:
                    split[i] = "-num({})-".format(self.get_name(Get(split, i)) + " " + Get(split, i+1) + " "+self.get_name(Get(split, i+2)))
                    split[i+1], split[i+2] = None, None

                continue

            iter_num = 1
            word = split[i-iter_num]

            while word in childtree:
                childtree = childtree[word]
                iter_num += 1
                word = split[i-iter_num]

            if childtree[None]:
                revised_word = ""
                for iternum in range(1, iter_num):
                    revised_word = split[i-iternum] + " " + revised_word
                    split[i-iternum] = None
                revised_word += self.get_name(split[i])
                split[i] = "-num({})-".format(revised_word)
                
            
               
         
            
        new_split = list(filter(lambda t : t and t != " ", split))
        
        
        return new_split, None
    
        
    
    
    ####################################################################################################
    def find_chemical_form(self, chem_name, tag = 'strange'):
        
        suffix_list = r"|".join(['powder', 'crystal', 'crystalline', 'particle', 'hollow', 'sphere', 'film', 'web', 'sheet', 'flower', 'fiber', 
                                 'atom', 'pore', 'composite', 'N[A-Z]s'])
        
        chem_name_revise = re.sub(fr"(?P<ABB_name>{self._ABB_re})", lambda t: self._ABB1[t.group("ABB_name")], chem_name)
        
        end_group = r"|".join(['type', 'based', 'layered', 'coated', 'decorated', 'activated', 'deposited', 'combined', 'supported',
                                              'ion', 'cation', 'anion', "N[A-Z]s", 'metal', 'metallic', 'bimetallic'])
        
        front_group = r"|".join(['pristine', 'bare', 'bared', 'pure', 'modified', 'stablized', '1D', '2D', '3D', '0D', '1-D', '2-D', '3-D',
                                 '0-D', 'metal', 'metallic', 'bimetallic'])
        
        chem_name_revise = re.sub(fr"(?<=\S+)-(?P<pos>{end_group}|{suffix_list}|\S+ed)s?(?=\s|$)|(?<=^|\s)(?P<pos>{front_group}|\S+ed)-(?=\S+)",
                                  lambda t: " {} ".format(t.group("pos")), chem_name_revise)
        
        chem_iter = chem_name_revise.split()
        
        #print (chem_iter)
        
        output = {'chemical': [''], 'chem_type': []}
    
        target = 'chemical'
        chem_sub_activation = True
        
        for chem in chem_iter:
            
            if chem in ['in', 'on', 'to', 'with']:
                #target = 'sub_material'
                output['chemical'].append('')
                
            elif chem in ['poly']:
                 output['chemical'][-1] += f" {chem}"
                    
            elif re.match(fr"({end_group}|{front_group})", chem):
                
                output['chemical'].append('')
                output['chem_type'].append(chem)
                
            elif re.match(fr"({suffix_list})(s|es|ies)?", chem) or re.match(r"^(nano|micro|macro|meso)\S+", chem):
                output['chemical'].append('')
                output['chem_type'].append(chem)
                
            elif not self.is_vaild_chemical(chem, tag, middle_possible = True):
                output['chemical'].append('')
                output['chem_type'].append(chem)
                
            elif re.match(r".+[@/]", chem):
                chem_split = re.sub(r"(?=.+)((?P<split>@)|\d/\d|(?P<split>/))", lambda t: " " if t.group("split") else t.group(0), chem).split()
                if len(chem_split)-1:
                    for chem_s in chem_split:
                        output['chemical'].append('')
                        output['chemical'][-1] += f' {chem_s}'
                    #output['chemical'][-1] += f' {chem_split[0]}'
                    #output['chemical'].append('')
                    #output['chemical'][-1] += f' {chem_split[1]}'
                    
                else:
                    #output[target] += chem_split[0]                
                    output['chemical'][-1] += f' {chem}'
            else:
                output['chemical'][-1] += f" {chem}"
        
        #output['chemical'] = output['chemical'].strip()
        #output['sub_material'] = output['sub_material'].strip()
        
        output['chemical'] = [word.strip() for word in output['chemical'] if word]
        
        return output
    
    def is_vaild_chemical(self, name, tag = 'strange', middle_possible = False):
        if name not in self._chemname and name not in self._strange and name not in self._ion1 and name not in self._sm1:
            if name in self._ABB1 and self._ABB2[self._ABB1[name]].ABB_type: # ABB
                pass
            else:
                return False
        if re.search(r"_\[.+\]", name):
            return False
        
        elif not middle_possible and re.findall(r"(yl|lic)$", name):
            return False
        
        elif re.search(r"(ing|ed)$", name): # Alkyl, caboxylic, N2-sorting, N2-sorpted
            return False
                    
        elif re.match(r"^(?i){}(s|es)$".format(r"|".join(self.MOLECULAR_NAMES)), name):
            
            return True

        elif not middle_possible and self._remove_not_chem_group(name):
            return False

        #elif re.match(r"^({element})((⋯|-)({element}))+$".format(element = r"|".join(self.ELEMENT)), name): 
        #    new_split[i] = name

        elif re.match(r"^([A-Za-z]/[A-Za-z])0?$", name): # P/P0, Z/Z0 
            return False

        elif re.search(r"[Ff]igure|[Tt]able|[Ff]ig|ESI", name): #Figure2
            return False

        elif re.search(r"\S+[-](type|fold|storage)(s|es)?", name): # N2-type / 3D-fold / Li-storage
            return False

        elif tag == 'strange':
            #if re.match(r"^[a-z]+(-[a-z]+)+(\s[a-z]+(-[a-z]+)+)*$", name):
            #    new_split[i] = name

            if re.match(r"^[A-Za-z][a-z]*[.]$", name): # Word. / word.
                return False

            elif re.match(r"[A-Za-z][a-z]*([-/\s][A-Za-z][a-z]*)+$", name):
                activation_s = False
                split_words = re.split(r"[/-]",name)
                for split_word in split_words:
                    if split_word in self._chemname or split_word in self._sm1 or split_word in self.ELEMENTS_AND_NAMES:
                        activation_s = True
                if activation_s:
                    #self._chemical_counter.update([name])
                    return True
                else:
                    return False
                
            elif self._unit_database.is_unit(name):
                #print ('unit', name, " ".join(self.sent))
                return False
            
        return True
        
    def _remove_not_chem_group(self, word):
        functional = ["alkane", 'alkene', 'alkyne', 'haloalkane', 'fluoroalkane', 'chloroalkane','bromoalkane','iodoalkane', 
                      'alcohol', 'ketone', 'aldehyde', 'acyl halide',
                       'carbonate', 'carboxylate', 'carboxylic acid', 'ester', 'methoxy','hydroperoxide',
                       'peroxide', 'ether', 'hemi-?acetal', 'hemi-?ketal', 'acetal', 'ketal', 'orthoester', 
                       'heterocycle','orthocarbonate ester', 'organic acid anhydride', 'amide', 'amine',
                       'imine','imide','azide', 'azo compound', 'cyanate', 'isocyanate', 'nitrate', 'nitrile',
                       'nitro compound', 'nitroso compound', 'oxime', 'carbamate', 'carbamate ester',
                       'thiol', 'sulfide', 'thioether', 'disulfide', 'sulfoxide', 'sulfone', 'surfinic acid',
                       'surfonic acid', 'sulfonate ester', 'thiocyanate', 'thioketone', 'thial', 'thiocarboxylic acid',
                       'thioester', 'dithiocarboxylic acid', 'dithiocarboxylic acid ester', 'phosphine',
                       'phosphonic acid', 'phosphate', 'phosphodiester', 'boronic acid', 'boronic ester',
                       'borinic acid', 'borinic ester', 'alkyllithium', 'alkylmagnesium halide', 'alkylaluminium',
                      'silyl ether', 'oxide', 'carbide', 'metal', 'semi-metal', 'carbon nitride', 'enol']
        
        not_chemical = ["micropore", 'macropore', 'meso', 'micro', 'macro','mesopore','estimated', 'trend', 'discovery', 'bore', 
                        'retain', 'backbone', 'scene', 'lamps', 'zip', 'ribbon', 'school', 'join', 'cyan', 'theme', 'plugin',
                        'ramp', 'doc', 'constant', 'constance', 'poly', 'polymer', 'estimate', 'pure', 'bare', 'pristine', 
                        'exfoliate', 'polymeric', 'visible-light', 'metal', '1D', '2D', '3D', 'measuring', 'pH', 'reforming']
        
        return re.match(r"(?i)^(?:{func}|{nocem})(?:s|es)?$".format(func=r"|".join(functional), nocem = r"|".join(not_chemical)), word)

    def unit_num_pair(self, split):

        unit_num_pair = []
        unit_index = None
        activation = False
        lens = len(split)-1

        for i, token in enumerate(split[::-1]):
            if self.identify_tag(token) == 'unit':
                activation = True
                unit_index = lens - i
            elif activation and self.identify_tag(token) == "number":
                unit_num_pair.append((unit_index, lens - i))
            elif activation and token not in [",", "and", "or", "vs", "vs."]:
                activation = False
                unit_index = None

        return unit_num_pair    
    
    def find_all_chemical(self, cut_off = 0):
        """Find all of chemical used in the paper
        document = DocumentTM('paper_path')
        set_chemicals = document.find_all_chemical(cut_off = 0)

        # Arguments
            cut_off : minimum number of chemical used in the paper ( Default = 0 )

        # Returns
            A set of chemicals that used in the paper
            {MOF-5, 'MIL-101', 'acetic acid', ....}

        """
        if not self._chemical_counter:
            self.doc()
            self.find_strange()
            for para in self.Para:
                new_split, unit = self.tokenize_paragraph(para, lemma=False, Strange=True)

        chem_word = []
        for cem, num in self._chemical_counter.items():
            if num > cut_off:
                chem_word.append(cem)
        return chem_word
            


# In[9]:


if __name__ == '__main__':
    
    #Q._strange = Counter(['Cu-N', 'Zn/L', 'P/P0'])
    with open("./OLED/abb_list_OLED.json", 'r') as f:
        abb_list = json.load(f)
        ABB1, ABB2 = make_abbreviation(abb_list)
        database = {'ABB1':ABB1, 'ABB2':ABB2}
        
    Q = DocumentTM("t.html", **database)
    
    text = """for real T_[d], T_[d], T_[d], TdTd"""
    Q.find_strange(text = text, cut_off=0)
    print (Q._strange)
    
    float_re = r"[+-]?\d+\.?\d*"
    special_unit_dictionary = {'tuple':fr"(?P<NUM>\(\s?({float_re})\s?,\s?({float_re})\s?\))"}
    Q.set_special_unit(special_unit_dictionary)
    
    print (Q.paragraph_preprocessing(text))
    
    B = Q.tokenize_test(text)
    A = Q.tokenize_paragraph(text, cut_off=False, lemma=False, Strange = True)
    
    
    print (A[0])
    print (B)
    print ("\n")
    #for chem in Q._chemical_counter:
    #    print (chem)
    #    print (Q.find_chemical_form(chem))
    #    print ("\n")


# In[28]:


if __name__ == "__main__":
    
    file = './OLED/test_pdf/1.html'
    #file = "./Core_MOF/html/IXOFAD-ic1025087.html"
    Q = DocumentTM(file)
    #Q = DocumentTM("Catalyst_paper/wiley_467.html")
    #Q = DocumentTM("Catalyst_paper/wiley_3528.html")
    Q.doc()
    float_re = r"[+-]?\d+\.?\d*"
    special_unit_dictionary = {'tuple':fr"(?P<NUM>\(\s?({float_re})\s?,\s?({float_re})\s?\))"}
    Q.set_special_unit(special_unit_dictionary)
    
    Q.find_strange()
    for para in Q.Para:
        try:
            #new_split, unit = Q.tokenize_paragraph(para, lemma = False, Strange = True, cut_off = False)
            new_split, _ = Q.tokenize_test(para)
        except:
            continue
        if new_split:
            print (para)
            print (" ".join(new_split))
            print ("\n\n")
            
            pass
    #print (Q._strange, Q._chemname.keys())
    
            
            
    #print (list(filter(lambda x y: y > 1, Q._chemical_counter.items())))
    #print (Q._sm1)
    #print ([(i, str(j), j.ABB_type) for i, j in Q._ABB2.items()])


# In[ ]:


if __name__ == '__main__':
    direc = Path("./OLED/test_pdf/")
    files = direc.glob("*.html")
    for i, file in enumerate(files):
        if i>100:
            break
        Q = DocumentTM(file)
        #Q = DocumentTM("Catalyst_paper/wiley_467.html")
        #Q = DocumentTM("Catalyst_paper/wiley_3528.html")
        Q.doc()

        Q.find_strange()
        for para in Q.Para:
            new_split, unit = Q.tokenize_paragraph(para, lemma = False, Strange = True, cut_off = False)
            #print (" ".join(new_split))


# In[ ]:


def find_Abstract_ACS(file_name, return_only_paragraph = False, **database):
    """Find Abstract from ACS paper
    >> input : 
    file_name : name of file
    return_only_paragraph : if True, return only abstract. Else, return abstract and DocumentTM
    
    output : <cde.paragraph>, <DocumentTM>
    """
    
    Q = DocumentTM(file_name, **database)
    Q.doc()
    
    Q.find_strange()
    
    
    with open(file_name, 'rb') as f:
        doc = Document.from_file(f)
        
    paras = list(filter(lambda t: isinstance(t, Paragraph) or isinstance(t, Heading), doc.elements))
        
    
    activation = False
    #for para in Q.Para:
    for para in paras:
        #print (para)
        if re.findall("(?i)abstract", para.text):
            return True
    return False
        
        
        
    """    if activation:
            if return_only_paragraph:
                return para
            else:
                return para, Q
            
        elif re.match(r"Share a link",para.text):
            activation = True
            
    if return_only_paragraph:
        return None
    else:
        return None, Q"""


# In[ ]:


def _find_from_dict(q):
    if isinstance(q, list):
        for q_sub in q:
            for j in _find_from_dict(q_sub):
                yield j
    elif isinstance(q, dict):
        if '_' in q:
            yield q['_']
        else:
            for q_sub in q.values():
                for j in _find_from_dict(q_sub):
                    yield j
                    
def find_Abstract(file):
    with open(file, encoding = 'UTF-8') as f:
        bs = BeautifulSoup(f, 'html.reader')
        
    z1 = bs.find_all('div', 'article-section__content en main')+bs.find_all('div', id='Abs1-content')
    z2 = bs.find_all('script', type='application/json')
    if z2:
        v = json.loads(z2[0].text)
        z = v.get('abstracts').get('content')
        if not z:
            print ('Error', file)
        else:
            return " ".join(_find_from_dict(z))
    else:
        for tag in bs.find_all('div')+bs.find_all('p'):
            if re.findall(r"(?i)(?<=^|[^a-z])(abstract|abs[^a-z])", " ".join(tag.get('class', []))):
                return tag.get_text()
            
            elif tag.id and re.findall(r"(?i)(?<=^|[^a-z])(abstract|abs[^a-z])", " ".join(tag.id)):
                return tag.get_text()
            
    if z1:
        return z1[0].get_text()
        
            
    
if __name__ == '__main__':
    direc = Path("./Abstract_chemical/electrode_html/")
    files = direc.glob("*.html")
    for file in files:
        z =  find_Abstract(file)
        if not z:
            print (file)


# In[ ]:


if __name__ == '__main__':
    direc = Path("./Battery_paper//")
    files = direc.glob("*.html")
    for i, file in enumerate(files):
        if i>100:
            break
        Abs, Q = find_Abstract_ACS(file, return_only_paragraph=False)
        
        new_split, _ = Q.tokenize_paragraph(Abs, lemma = False, Strange = True)
        if not new_split:
            continue
        
        chem_iter = filter(lambda t: re.match(r"-c\d{5,7}-", t), new_split)
        chem2_iter = filter(lambda t: re.match(r"-strange\(\S+\)-", t), new_split)
        ion_iter = filter(lambda t: re.match(r"-i\d{5,7}-", t), new_split)
        sm_iter = filter(lambda t: re.match(r"-s\d{5,7}-", t), new_split)
        
        iter_iter = [chem_iter, chem2_iter, ion_iter, sm_iter]
        chem_list = [[], [], [], []]
        
        for i, iters in enumerate(iter_iter):
            for chem in iters:
                chem_original = word_original(chem, **Q.database())
                chem_list[i].append(chem_original)
        
        print ("---------------------------")
        print (file)
        print (Abs)
        #print (" ".join(new_split))
        
        #for word, cl in zip(['cem', 'str','ion', 'sm'], chem_list):
        #    print (f'{word}: ', set(cl))
            
        for word, cl2 in zip(['chemical', 'strange','ion', 'smallmolecule'], chem_list):
            for cl in set(cl2):
                print (cl)
                print (Q.regularize_chemical(cl, word))


# In[ ]:





# In[ ]:




