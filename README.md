# MOFtextminer : Text mining tools for extracting synthetic methods in MOF



## 1. class MofDictionary

### 1) Create class < MofDictionary > from file (xml, html, pdf)

> ```python
> from mofdict import MofDictionary
> from reader import GeneralXmlReader
> 
> output = MofDictionary.from_file(paragraph, reader=GeneralXmlReader)
> ```

- filepath : path of file (str, bytes, os.PathLike, or pathlib.Path)
- Reader : parser of file (The appropriate reader must be selected.)

  - GeneralXmlReader : Xml reader for ACS and Springer journal

  - ElsevierXmlReaer : Xml reader for Elsevier journal
  - CDEXmlReader : Xml reader using ChemDataExtractor parser

  - CDEHtmlReader : Html reader using ChemDataExtractor parser
  - CDEPdfReader :  PDF reader using ChemDataExtractor parser
- standard_unit (default = True) : If True, unit of conditions and properties are unified to SI Unit.
- find_abbreviation (default = False) : If True, find abbreviation in text (Slow processing). 
- convert_precursor (default=False): If True, metal precursor would be convert to chemical formula, organic precursor would be convert to smiles, and extra precursor would be convert to specific solvent precursor.



### 2) Create class < Mof Dictionary > from output file

```python
mof = MofDictionary.from_dict(output_dictionary) # create from dictionary

mof = MofDictionary.from_json(json_file_path) # create from json file
```



### 3) Instance of class < MofDictionary>

```python
mofdict.mof_list  # list of class <MOF>

mofdict.raw_elements  # paragraph of MOF paper
mofdict.elements # clean-up paragraph of MOF paper
mofdict.raw_method_paragraphs  # method paragraph of MOF paper
mofdict.method_paragraphs  # clean-up method paragraphs of MOF paper

mofdict.abbreviation  # abbreviation of MOF paper

mofdict.metadata  # metadata of MOF paper
mofdict.title  # title of MOF paper
mofdict.doi  # doi of MOF paper
mofdict.url  # url of MOF paper
mofdict.journal  # journal of MOF paper
mofdict.author_list  # author list of MOF paper
mofdict.date  # date of MOF paper
```



### 4) Function of class < MofDictionary >

```python
mofdict.to_dict   # save mofdictionary to dictionary
```





## 2. class MOF

### 1) Create class < MOF > from synthesis paragraph

> ```python
> from mof import MOF
> from mofdict import MofDictionary
> 
> mofdict = MofDictionary.from_file(file_path) # create MOF from MofDictionary
> mof_list = mofdict.mof_list # list of class <MOF>
> mof = mof_list[index]
> 
> mof = MOF.from_paragraph(paragraph) # create MOF directly from paragrpah
> ```



### 2) Create class < MOF > from output

> ```python
> output_dict = {'name': '1',
> 'symbol': None,
> 'M_precursor': [{'name': 'Cu(NO3)2·3H2O',
>  'composition': [('48.3', 'mg1.0'), ('0.2', 'mmol1.0')]}],
> 'O_precursor': [{'name': '1,4-H2ndc',
>  'composition': [('86.5', 'mg1.0'), ('0.4', 'mmol1.0')]},
> {'name': 'EMIM-l-lactate',
>  'composition': [('100', 'mg1.0'), ('5', 'mmol1.0'), ('25', 'mL1.0')]}],
> 'S_precursor': [],
> 'temperature': {'Value': '140', 'Unit': '°C1.0'},
> 'time': None}
> 
> mof = MOF.from_dict(output_dict)  # create from dictionary
> 
> mof = MOF.from_json(josn_file_path) # create from json_file
> ```



### 3) Instance of class < MOF >

#### Main instance

> ```python
> mof.name  # name of MOF
> mof.symbol  # symbol of MOF
> mof.temperature  # represent temperature of MOF
> mof.time  # represent time of MOF
> mof.M_precursor  # metal precursor of MOF
> mof.O_precursor  # organic precursor of MOF
> mof.S_precursor # extra precursor of MOF
> mof.MOratio # M/O ratio of MOF
> mof.doi # Doi
> mof.operation # operation of MOF synthesis
> mof.method # Method of MOF synthesis 
>            #(conventional solvothermal, microwave, sonochemical, electrochemical, mechanochemical)
> ```



#### hidden instance

> ```python
> mof._precursor # all precursors of MOF
> mof._target # all targets of MOF
> mof._etc # all etc material of MOF
> mof._text # paragraph of mof
> ```



### 4) Function of class < MOF >

#### MOF.to_dict

- extract_all : if True, extract all instance (include hidden instance)

> ```python
> # to_dict function
> mof.to_dict()
> 
> # If want to extract all instance
> mof.to_dict(extract_all = Ture)
> ```



#### MOF.append_linker

- material : (dictionary) name and composition of material
- astype : (str) 'precursor', 'target', and 'etc'

> ```python
> new_precursor = {'name': 'CuO2',
>                  'composition': [('4', 'mg1.0'), ('2', 'mmol1.0')]}
> mof = MOF.from_dict(dict_)
> 
> # append material using dictionary
> mof.append_material(new_precursor, 'precursor')
> 
> # append material using name and composition
> mof.append_material(name='CuO2', composition=[('4', 'mg1.0')], astype='precursor')
> 
> # append target
> mof.append_material(name='2', astype='target')
> ```



#### MOF.remove_linker

- material : (dictionary) name and composition of material
- name : (str) name of material. if there are no composition, remove all material having same name.

> ```python
> remove_precursor = {'name': 'CuO2',
>             	        'composition': [('4', 'mg1.0'), ('2', 'mmol1.0')]}
> mof = MOF.from_dict(dict_)
> 
> # remove material using dictionary
> mof.remove_material(remove_precursor)
> 
> # remove material using name and composition
> mof.remove_material(name='CuO2', composition=[('4', 'mg1.0'), ('2', 'mmol1.0')])
> 
> # remove material using name (remove all material with same name)
> mof.remove_material(name='CuO2')
> 
> # remove target
> mof.remove_material(name='1')
> ```



#### MOF.get_material_list

- list of material names in MOF
- attribute : M_precursor, O_precursor, S_precursor, target, precursor, etc

> ```python
> # list of metal precursors
> mof.get_material_list('M_precursor')
> 
> # list of precursors
> mof.get_material_list('precursor')
> 
> # list of targets
> mof.get_material_list('target')
> ```



#### ~~MOF.classify_matrial~~

~~After MofDict processes replace, proceed automatically. (No need to use)~~

> ```python
> # if use function classify_material, M, O, S precursors are saved
> mof.classify_material()
> ```





### 5) Operation

It can be checked through the instance `operation` of MOF objects.

> ```python
> mof.operation
> ```

Format : List (elements : dictionary)



- Each operation (dict) is stored in the list in order.
- Operation has the keys 'name' and 'condition'.
- Condition stores each property (dict). Each property has a key of 'Value', 'Unit' and 'Property'.

> ```json
> [{'name': 'stir',
>   'condition': [{'Value': '2', 'Unit': 'h1.0', 'Property': 'Time'},
>    {'Value': '80', 'Unit': '°C1.0', 'Property': 'Temperature'}},
>  {'name': 'remove',
>   'condition': [{'Value': '293', 'Unit': 'K1.0', 'Property': 'Temperature'},
>    {'Value': '1', 'Unit': 'atm1.0', 'Property': 'Pressure'}]},
>  {'name': 'dissolve', 'condition': []},
>  {'name': 'wash', 'condition': []},
>  {'name': 'dry', 'condition': []},
>  {'name': 'remove', 'condition': []}]
> ```

The types of operations and keywords that can be saved are as follows.

(..) means that a variety of utilization types are available. (ex. heat(..) : heat, heating, heated, ...)

- Heat : heat(..), oven, autoclave, Teflon-lined, solvothermal, hydrothermal
- Cool : cool(..), refrigerator
- Stir : stir(..)
- Wash : wash(..)
- Remove : remov(..)
- Dehydrate : dehydrat(..)
- Desicate : desicat(..)
- Dissolve : dissolv(..), redissolv(..)
- Sonicate : sonic(..), ultrasonic(..)
- Diffuse : diffus(..)
- Store : stor(..)
- Wait : wait(..), left(..), keep(..), kept
- Purify : purif(..)
- Linse : lins(..)
- Filter : Filter(..)
- Dry : dri(..). dry(..)
- Ground : ground(..)
- Evaporate : evaporat(..)
- Crystallize : crystalliz(..), recrystalliz(..)



### 4. Property

It can be checked through the properties of MOF objects.

> ```python
> mof.property
> ```

Format : Dictionary



- Each property (dict) is stored in the dictionary.
- Each property has a key of 'Value' and 'Unit'.

> ```json
> {'yield': {'Value': '81', 'Unit': '%1.0'},
>  'Melting point': {'Value': '220-222', 'Unit': '°C1.0'}}
> ```

The properties that can be saved are as follows.

- Melting point
- Boiling point
- Decomposition temperature



# 3. class AccuracyResult

Class for measuring accuracy of precursor, Mof, and MofDictionary.



It can import inheritance classes for functions compare and class AccuracyResult from evaluation.

> ```python
> from evaluation import compare, MofdictionaryAccuracyResult, MofAccuracyResult, PrecursorAccuracyResult
> ```



### Example of how to use (automate comparison code with answer json)

> ```python
> result = MofdictionaryAccuracyResult() # Create Class for Accuracy Measurement
> path = Path('example/Testset_answer/')
> for i, file in enumerate(path.glob("*.json")):
>     try:
>         file_ = file.stem.replace("ans_", "")
>         file_ = Path(f'example/ACS_XML/{file_}.xml')
>         
>         mofdict = MofDictionary.from_file(file_, GeneralXmlReader) #Generate Result mofdict
>         mofdict_answer = MofDictionary.from_json(file) # Actual answer (loading saved json file)
>         result.compare(mofdict, mofdict_answer) # Compare two correct answers, save to result
>         
>     except KeyboardInterrupt:
>         break
>     except Exception as e:
> 		pass
> ```


