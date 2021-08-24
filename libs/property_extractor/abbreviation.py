#!/usr/bin/env python
# coding: utf-8

# In[3]:


from functools import reduce

import numpy as np
import regex as re

from fuzzywuzzy import fuzz


# In[7]:


class Abbreviation():
    class Counter_temp():
        def __init__(self, ABB_type=None, Trainable = True):
            self.count = np.zeros(2, dtype = 'int16')
            self.Trainable = True
            self.update(ABB_type, Trainable)
            
        def __repr__(self):
            return "({}/{})".format(self.count[0], self.count[1])
            
                
        def update(self, ABB_type=None, Trainable = True):
            if not Trainable:
                self.Trainable = False
                self.ABB_type = ABB_type
                if ABB_type:
                    self.count = np.array([0,-1], dtype = 'int8')
                else:
                    self.count = np.array([-1,0], dtype = 'int8')
                
            if not self.Trainable:
                return self.ABB_type
                
            if isinstance(ABB_type, list):
                for types in ABB_type: self.update(types)
            
            if self.checking(ABB_type):
                self.count[0] += 1
            else:
                self.count[1] += 1
                
            self.ABB_type = self.ABB_type_checking()
            return self.ABB_type
        
        def ABB_type_checking(self):
            label = ["CM", None]
            return label[np.argmax(self.count)]
            
        def checking(self, types = None):
            if types == "CM":
                return True
            else:
                return False
            
    def __init__(self, ABB_name, ABB_def, ABB_type_original = None, Trainable = True): 
        self.ABB_def = ABB_def
        self.ABB_name = ABB_name
        
        self.ABB_type = None
        self.ABB_class, self.ABB_class_type = [], []
        self.update(ABB_def, ABB_type_original, Trainable = Trainable)

        return None
    
    def _check_ABB_type(self, ABB_name, ABB_def, ABB_type=None):
        
        checking_string = ABB_def
        checking_string = re.sub(r"\b-\b", " - ", checking_string)
        checking = Paragraph(checking_string).cems
        
        if checking:
            ABB_type = 'CM'
        else:
            ABB_type = None
            
        return ABB_type
    
    def _check_validation(self, ABB_def, ABB_type):
        ABB_name = self.ABB_name
        ABB_front_char = reduce(lambda x, y: x + "".join(re.findall(r"^\S|[A-Z]", y)), re.split(r",|\s|-", ABB_def), "")
        if ABB_name[-1] == 's' and ABB_def[-1] == 's':
            ABB_front_char += 's'
        ratio = fuzz.ratio(ABB_name.lower(), ABB_front_char.lower())
        
        return ratio > 70 or ABB_type == 'CM'
    
    def __repr__(self):
        return "("+") / (".join(self.ABB_class)+")"
    
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
    
    
    def get(self, key, default = None):
        try:
            return self[key]
        except:
            return default
            
    
    def change_ABB_type(self, ABB_def, ABB_type):
        self.update(ABB_def, ABB_type, Trainable = False)
    
    def update(self, ABB_def, ABB_type_original = None, Trainable = True):
        ABB_type = ABB_type_original
        
        if isinstance(ABB_def, list):
            for defs in ABB_def: self.update(defs, ABB_type)
            return
        
        #print (self.ABB_class, ABB_def)
        for i, classification in enumerate(self.ABB_class):
            result = self.compare(ABB_def, classification)         
            if result > 70:
                self.ABB_class_type[i].update(ABB_type, Trainable=Trainable)
                
                self.ABB_def = classification
                self.ABB_type = self.ABB_class_type[i].ABB_type
                
                return None
            
        self.ABB_class.append(ABB_def)
        
        self.ABB_class_type.append(self.Counter_temp(ABB_type, Trainable))
        
        self.ABB_def = ABB_def
        self.ABB_type = ABB_type

    def compare(self, text1, text2, re1 = False, x1 = False):
        #print (text1, text2, fuzz.ratio(text1.lower(), text2.lower()))
        return fuzz.ratio(text1.lower(), text2.lower())

def make_abbreviation(ABB_list, ABB_dictionary1 = None, ABB_dictionary2 = None, Trainable = False):
    """ABB1, ABB2 = make_abbreviation([('ASAP', 'as soon as possible', None), ('DKL', 'depolymerised Kraft lignin', 'CM')], ABB1, ABB2)
    
    ABB_list : List of tuple, (ABB_name, ABB_definition, ABB_type)
    ABB_dictionary1 : Dictionary (key : ABB_definition, value : ABB_name)
    ABB_dictionary2 : Dictionary (key : ABB_name, value : class <abbreviation> )
    
    update and return ABB_dictionary1 and ABB_dictionary2 that append abbreviations

    """
    
    if not isinstance(ABB_dictionary1, dict):
        ABB_dictionary1 = {}
        
    if not isinstance(ABB_dictionary2, dict):
        ABB_dictionary2 = {}
    
    for ABB_name, ABB_def, ABB_type in ABB_list:
        if ABB_name in ABB_dictionary2:
            new_abb = ABB_dictionary2[ABB_name]
            new_abb.update(ABB_def, ABB_type, Trainable)
            ABB_dictionary1[ABB_def] = ABB_name

        else:
            new_abb = Abbreviation(ABB_name, ABB_def, ABB_type, Trainable)
            if not len(new_abb):
                return None, None
            ABB_dictionary2[ABB_name] = new_abb
            ABB_dictionary1[ABB_def] = ABB_name
            ABB_dictionary1[ABB_name] = ABB_name
        
    return ABB_dictionary1, ABB_dictionary2


# In[8]:


if __name__ == '__main__':
    
    from chemdataextractor.doc import Paragraph
    abb_list = Paragraph("hydrogenated V2O5\u2020 (H-V2O5\u2020) nanosheets").abbreviation_definitions
    print ('ABB_list : ', abb_list)
    for abb in abb_list:
        ABB, ABB_def, ABB_type = abb
        ABB = " ".join(ABB)
        ABB_def = " ".join(ABB_def)
        z = Abbreviation(ABB, ABB_def, ABB_type)
        print (z, z.ABB_type)
        
        ABB_front_char = reduce(lambda x, y: x + "".join(re.findall(r"^\S|[^a-z]", y)), re.split(r",|\s|-", ABB_def), "")
        if ABB[-1] == 's' and ABB_def[-1] == 's':
            ABB_front_char += 's'
            
        ABB_char = re.sub(r"(,|\s|[-])", "", ABB)
        
        print (ABB_char.lower(), ABB_front_char.lower())
        ratio = fuzz.ratio(ABB_char.lower(), ABB_front_char.lower())
        print (ratio)


# In[6]:


if __name__ == '__main__':
    import json
    with open("./abb_list.json", 'r') as f:
        abb_list = json.load(f)
    ABB1, ABB2 = make_abbreviation(abb_list)


# In[ ]:




