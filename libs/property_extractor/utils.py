#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pickle

import regex as re
from pathlib import Path


# In[10]:


def Get(target, index, default = None):
    """return target[index] when index is in range of [0, len(target)-1],
            default when index is out of range (default = None)
            
    """
    if not isinstance(target, list) and not isinstance(target, str):
        raise TypeError("function <Get> receives only list")
    
    try:
        if isinstance(index, tuple):
            st, ed = index
            value = target[st:ed]
        elif isinstance(index, int):
            value = target[index]
            
        return value
    
    except IndexError:
        return default
    
def is_list_in(sent, target):
    """ return True when the target is in sent
    is_list_in([1,2,3,4], [2,3]) -> True
    is_list_in([1,2,3,4], [2,4]) -> False"""
    
    for j, word in enumerate(sent):
        if word == Get(target, 0):
            activation = True
            for i, word_compare in enumerate(target):
                if word_compare != Get(sent, j + i):
                    activation = False
                    break
            if activation:
                return True
    return False

def contain_element(large_list, small_list):
    """ return True when large_list contains elements in small_list
    contain_element([1,2,3], [2,4]) -> True
    contain_element([1,2,3], [4,5]) -> False"""
    
    for word in small_list:
        if word in large_list:
            return word
    return False


# In[2]:


def Define(target, default, func = None):
    """return func(target) if bool(target) is True
    return default if bool(target) is False
    
    >>> a = None
    >>> value = Define(a, 0, lambda t: t+1)
    >>> value
    0
    
    >>> b = 2
    >>> value = Define(b, 0, lambda t: t+1)
    >>> value
    3
    """
    if not callable(func):
        func = (lambda t: t)
    if target:
        return func(target)
    else:
        return default
    
    
def Check(target, default, func = None):
    """return target if func(target) is True
    return default if func(target) is False
    
    >>> a = None
    >>> value = Define(a, 0, lambda t: t+1)
    >>> value
    0
    
    >>> b = 2
    >>> value = Define(b, 0, lambda t: t+1)
    >>> value
    3
    """
    
    if not callable(func):
        func = (lambda t: t)
        
    if func(target):
        return target
    else:
        return default
    


# In[3]:


def Update_list_in_dictionary(dictionary, key, *arg):
    """update list in dictionary. if there are no list, make list
    
    >>> z = Update_list_in_dictionary({'a':[1], 'b':[2]}, 'c', 0)
    >>> z
    {'a':[1], 'b':[2], 'c':[0]}
    
    >>>z = Update_list_in_dictionary({'a':[1], 'b':[2]}, 'a', 0)
    >>> z
    {'a': [1, 0], 'b': [2]}
    
    """
    if key not in dictionary:
        dictionary[key] = []
    dictionary[key].extend(arg)
    
        
    return dictionary


# In[4]:


def small_molecule_database(set_of_sm, sm_dict1 = None, sm_dict2 = None):
    """sm1, sm2 = small_molecule_database({H2, H2O}, sm1, sm2)
    set_of_sm : set of small_molecule
    sm_dict1 : Dictionary (key : small molecule, value : hashcode)
    sm_dict2 : Dictionary (key : hashcode, value : small molecule)
    
    update and return sm_dict1 and sm_dict2 that append abbreviations
    """
    if not isinstance(sm_dict1, dict):
        sm_dict1 = {}
    if not isinstance(sm_dict2, dict):
        sm_dict2 = {}

    
    sm_dict1.update({v:"-s{}-".format(str(i).zfill(6)) for i, v in enumerate(set_of_sm) })
    sm_dict2.update({"-s{}-".format(str(i).zfill(6)):v for i, v in enumerate(set_of_sm) })
    
    return sm_dict1, sm_dict2


# In[5]:


def hash_original(word, **database):
    
    tag = word[1]
    
    element = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K",
    "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I",
    "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr",
    "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf",
    "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og", "Uue"]

    if tag == 'c':
        return database.get('chemhash')[word]
    elif tag == 'i':
        return database.get('ion2')[word]
    elif tag == 's':
        return database.get('sm2')[word]
    elif tag == 'e':
        return element[int(word[2:-1])]
    elif tag == 'u':
        return database.get('unit_database')[word]['unit']

def word_original(sentence, **kwargs):
    """ ** kwargs
    chemhash : (dict) chemical hashcode to chemname
    ion2 : (dict) ion hashcode to name
    sm2 : (dict) smallmolecule hashcode to name
    unit_database : (Unit_database) unit database
    general : (bool) if True, hash changes as -cem-, -ion-, -smallmolcule-, -element-
    
    """
    general = kwargs.get('general', False)
    
    if not general:
        sent_copy = re.sub(r"([-](?:c|i|e|s|u)\d{5,6}[-])", string = sentence, 
                      repl = lambda z : hash_original(z.group(), **kwargs))
        sent_copy = re.sub(r"-(strange|num)[(](?P<name>\S+|.+?)[)]-", string=sent_copy, repl = lambda t: t.group("name"))
        sent_copy = re.sub(r"[(](bold|italic)[)]", "", sent_copy)
        return sent_copy
        
    else:
        general_dict = {'c':'-cem-', 'i':'-ion-', 'e':'-element-', 's':'-smallmolecule-', 'u':'-unit-'}
        sent_copy = re.sub(r"[-](c|i|e|s|u)\d{5,6}[-]", string = sentence, repl = lambda z :general_dict[z.group(1)])
        sent_copy = re.sub(r"-(num)[(](?P<name>\S+|.+?)[)]-", '-num-', sent_copy)
        sent_copy = re.sub(r"-(strange)[(](?P<name>\S+|.+?)[)]-", '-cem2-', sent_copy)
        sent_copy = re.sub(r"[(](bold|italic)[)]", "", sent_copy)
        return sent_copy
    
    
def sentence_original(sentence, **kwargs):
    """sentence_original(sentence, cc2, ion2, sm2, unitlist2) -> remove hash-tag 
    ** kwargs
    chemhash : (dict) chemical hashcode to chemname
    ion2 : (dict) ion hashcode to name
    sm2 : (dict) smallmolecule hashcode to name
    unit_database : (Unit_database) unit database
    general : (bool) if True, hash changes as -cem-, -ion-, -smallmolcule-, -element-
    islist: (bool) if True, return as list. else, return as iteration
    
    """
    islist = kwargs.get('islist', True)
    
    if isinstance(sentence, str):
        return word_original(sentence, **kwargs)
    
    if islist:
        return list(map(lambda word : word_original(word, **kwargs), sentence))
    
    else:
        return map(lambda word : word_original(word, **kwargs), sentence)


# In[6]:


def make_new_sentence(sentence_list, chemical_list, ion_list, sm_list, unit_database):
    new_sentence1, new_sentence2 = [], []
    
    for sent in sentence_list:
        list1 = []
        list2 = []
        for word in sent:
            list1.append(word_original(word, chemical_list, ion_list, sm_list, unit_database, True))
            list2.append(word_original(word, chemical_list, ion_list, sm_list, unit_database, False))
        new_sentence1.append(list1)
        new_sentence2.append(list2)
        
    return new_sentence1, new_sentence2


# In[7]:


def cut_paragraph(para):
    "input : Paragraph // output: list of sentence"
    new_list = [[]]
    index = 0
    for word in para:
        if word != '-end-':
            new_list[index].append(word)
        else:
            new_list.append([])
            index += 1
    return new_list

def cut_para_list(para_list, islist = False):
    "input : list of Paragraph // output: list of sentence"
    if islist:
        return list(reduce(lambda x, y: x + cut_paragraph(y), para_list, []))
    else:
        return reduce(lambda x, y: x + cut_paragraph(y), para_list, [])


# In[8]:


def save_dictionary(direc_save, global_dict, Dictname = ['unitanalysis', 'unitlist1', 'unitlist2', 'cc1', 'cc2', 'ABB1', 'ABB2',
        'sm1', 'sm2', 'ion1', 'ion2', 'total_Para']):
    "save_dictionary('direc', globals(), [dictionary])"
    
    path = Path(direc_save)
    if not path.exists():
        path.mkdir()

    for dictionary in Dictname:
        new_path = path/(dictionary+".pickle")
        text = """with open("{newpath}", 'wb') as file:
            global {name}
            pickle.dump({name}, file, pickle.HIGHEST_PROTOCOL)
        """.format(name=dictionary, newpath= new_path)
        exec(text, global_dict)
        
def load_dictionary(direc_save, global_dict, Dictname = ['unitanalysis', 'unitlist1', 'unitlist2', 'cc1', 'cc2', 'ABB1', 'ABB2',
        'sm1', 'sm2', 'ion1', 'ion2', 'total_Para']):
    "load_dictionary('direc', globals(), [dictionary])"
    for dictionary in Dictname:
        path = Path(direc_save)/(dictionary+".pickle")
        text = """with open('{path}', 'rb') as file:
            #global {name}
            {name} = pickle.load(file)
        """.format(name = dictionary, path = path)
        exec(text, global_dict)
        
        


# In[ ]:




