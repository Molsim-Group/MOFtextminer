import regex
from urllib.request import urlopen
from chemspipy import ChemSpider
from ase.formula import Formula

def cir_converter(ids, option='smiles', raise_error=False):
    """Change ids to option
    >>>
    ids : (str) chemical name 
    option : (str) smiles, formula"""
    assert option in ['smiles', 'formula']
    try:
        url = 'http://cactus.nci.nih.gov/chemical/structure/' + ids + '/'+ option
        ans = urlopen(url).read().decode('utf8')
        return ans
    except Exception as e:
        if raise_error:
            raise e
        else:
            return None
    
def chemspider_converter(ids, option = 'smiles', raise_error=False):
    """Change ids to option
    >>>
    ids : (str) chemical name 
    option : (str) smiles, formula"""
    
    #api_key = 'JYsy5GHJ94TwtmTjJDFKdVUK0g72ECGc' # YH
    #api_key = 'aZGBgCs1f0BAExqKN4n2o4nWAIR06sFl' # HS
    #api_key = '5nJLVbKcAsYMXv2xwVlqK0bE7ifHFpGg' #SH
    #api_key =  'LzaCAIhS17tlAWfG03g0GmqXCs3hz7BH' # JK
    #api_key = 'jwFfmXmB1G5dYytBQHfCXdiH4cfuWJuR' # HS2
    #api_key = 'D92aa2hH2qs4FNI0S4uSIt3ABUlp6O83' # HS3
    #api_key = 'uQPiN7O4RDDaS9pd1CR1g83YCOkjk9vy' # JW
    api_key = 'uGZVpziErVlmhK7hDtFdL8k3MgLwXDMk' # YH2
    api_key = 'NKf4jzeCWUuv9ePk6i9hCfLiKRd0Hlud' # YH3
    
    prx_list = {'mono':1, 'di':2, 'tri':3, 'tetra':4, 'penta':5, 'hexa':6, 'hepta':7, 'octa':8, 'nona':9, 'deca':10, 'undeca':11, 'dodeca':12, 'hemi':0.5, 'sesqui':1.5}
    prx_string = r"|".join(prx_list.keys())
    
    try:
        cs = ChemSpider(api_key)
    
        value = regex.match(fr"(?i)^(?P<cluster>.+?)(?:(?P<value>{prx_string})(?P<hydrate>hydrates?))?$", ids)
        cluster = value.group('cluster')
        value = value.group('value')
        
        c1 = cs.search(cluster)
        
        assert c1.exception is None, "Invailed key"
        
        for result in c1:
            if option == 'smiles':
                return result.smiles
            elif option =='formula':
                formula = result.molecular_formula
                text = regex.sub(r"_\{(.+?)\}", lambda t: t.group(1), formula)
                text = Formula(text).format('abc')
                
                if value:
                    val_ = prx_list[value.lower()]
                    text = text + f"${val_}H2O"
                    return text
                else:
                    return text
                
            else:
                raise TypeError()
        return None
    
    except Exception as e:
        if raise_error:
            raise e
        else:
            return None