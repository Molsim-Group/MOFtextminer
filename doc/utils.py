import regex

from chemdataextractor.doc import Paragraph

from error import DatabaseError


def cleanup_text(para):
    """Clean-up paragraph code
    :param para: (str or doc.Paragraph) Paragraph of paper
    :return: (str) clean up text
    """

    def remove_dot(sentence):
        output = sentence.text
        if output[-1] in ".!?":
            output = output[:-1]
        return output

    if isinstance(para, str):
        para = Paragraph(para)

    text = ' -end- '.join(map(remove_dot, para.sentences))
    text = text.replace('\n', " ")

    text = regex.sub(r"i\.e\.(?=,|\s)", "that is", text)  # remove i.e.
    text = regex.sub(r"e\.g\.(?=,|\s)", "for example", text)  # remove e.g.
    text = regex.sub(r"al\.,", "al. and", text)
    text = regex.sub(r"vs\s?\.", 'vs', text)  # change vs. -> vs

    # change unicode
    text = regex.sub(r"[\u2000-\u2005\u2007\u2008]|\xa0", " ", text)
    text = regex.sub(r"[\u2006\u2009-\u200F|\u00ad]", "", text)
    text = regex.sub(r"[\u2010-\u2015]|\u2212", "-", text)
    text = regex.sub(r"≈|∼", "~", text)
    text = regex.sub(r"\u2032|\u201B|\u2019", "'", text)
    text = regex.sub(r"\u2215", "/", text)
    text = regex.sub(r"\u201A", ',', text)
    text = regex.sub(r"\u201C|\u201D|\u2033", '"', text)

    text = regex.sub(r"\s?\b[.][,-]+\s?", " -end- ", text)  # remove ".-" or ".,," pattern
    text = regex.sub(r"\[\s([,-]\s)*\]", '', text)  # remove [ , , ]
    text = regex.sub(r"\[([,-])+\]", '', text)  # remove [,,]
    text = regex.sub(r"\b(([A-Za-z]{2,})[,.]\d+[A-Za-z]?)\b", lambda t: t.group(2) + " -end- ", text)
    text = regex.sub(r"(?<=\s)[.,-](?=\s)", " ", text)  # remove .,- between blanks
    text = regex.sub(r"\[\]|\(\)|{}", '', text)  # remove (), [], {}
    text = regex.sub(r"\u2022|\u2024|\u2027|\u00B7", r"\u22C5", text)


    # separate equality, inequality sign
    text = regex.sub(r"(\s?[\u221F-\u2281|\u2284-\u22C3\u22CD-\u22FF:;=><~]\s?)", lambda t: " {} ".format(t.group(1)), text)

    figure_table = r"(\b([Ff]igure|[Tt]able|[Gg]raph|[Ss]cheme|[Ff]ig)(s|es)?(?:(?:[,]|and|[.]|\s|[-]|\B)+(?:S?\d+|[A-Za-z]){1,2}\b(?:\s?[(][A-Za-z1-9][)]\B)?)+?)"
    text = regex.sub(figure_table, string=text, repl=lambda t: t.group(2))
    text = regex.sub(r"(([(])?(Fig|Table|Figure|ESI)\s?†(?(2)[)]))", string=text, repl="")

    # orbital (3p1/2 -> p'orbital)
    text = regex.sub(r"([1-9]([spdfg])\s?[13579]\s?[/]\s?2)", lambda t: "{}'orbital".format(t.group(2)), text)

    # type II curve -> -type-
    roman_num = r"\b(?:iv|ix|(?:v|x)i{0,3}|i{1,3})\b"
    types = r'(?i)type(\s-\s|-|\s)?(?P<b>\()?' + roman_num + r'(?(b)\))(\s-\s|-|\s)?(curve)?'
    text = regex.sub(types, " -type-", text)

    # Lee et al. -> -paper-
    text = regex.sub(pattern=r"(\s?\b\S+\set\.?\sal(\.)?(?(2)\B|\b)\s?)", string=text, repl=" -paper- ")

    # https://email@domain -> -email-
    text = regex.sub(r"((https?://)?\S+[.](com|net|org|edu)(/\S+)?)(?=\s|$)", " -email- ", text)

    # isotope with NMR
    text = regex.sub(r"(?P<atom>1H|2H|6Li|11B|13C|15N|19F|29Si|31P|195Pt)\s?\S*(nmr|NMR)",
                     lambda t: "{}-NMR".format(t.group("atom")), text)
    # .,:;
    text = regex.sub(r'(?P<A>(?<=\S+)[.,:;"](?=\S+))|(?P<OL>(?<=(^|\s|\(|\))[A-Za-z])[."](?=\s|\(|\)|$))|[.,:;"]',
                     lambda t: t.group() if t.group("A") or t.group("OL") else " {} ".format(t.group()), text)

    # lower first letter
    text = regex.sub(r"(?<=(^|-end-)\s*)(?P<word>[A-Z][a-z]+)(?=\s|$)", lambda t: t.group('word').lower(), text)

    # remove apostrophe s ('s)
    text = regex.sub(r"(?<=\S+)'s(?=\s|$)", " ", text)

    # separate bracket
    queue_bracket = []
    small_queue = []
    activation = True
    first_activation = False
    poly_activation = None

    for i, char in enumerate(text):
        if char in '({':
            small_queue.append(i)
            try:
                if text[i - 4:i] == 'poly':
                    poly_activation = i
                elif activation:
                    first_activation = True
                    activation = False
            except IndexError:
                pass

        elif char == " " and not poly_activation:
            activation = True
            first_activation = False

            if small_queue:
                queue_bracket += small_queue
                small_queue.clear()

        elif char in ')}':
            if small_queue:
                pop_index = small_queue.pop()

                if poly_activation and not (poly_activation - pop_index):
                    poly_activation = None

                elif not small_queue and first_activation:
                    try:
                        if text[i + 1] == ' ':
                            queue_bracket.append(pop_index)
                            queue_bracket.append(i)
                    except IndexError:
                        queue_bracket.append(pop_index)
                        queue_bracket.append(i)

                    first_activation = False
            else:  # Already removed
                queue_bracket.append(i)
        else:
            activation = False

    queue_bracket += small_queue

    revised_text = ""
    for i, char in enumerate(text):
        if i in queue_bracket:
            revised_text += " {} ".format(char)
        else:
            revised_text += char

    # Remove 'a' and 'the'
    revised_text = regex.sub(r"(?<=^|\s)(?:a|the|an)(?=\s|$)", " ", revised_text).strip()

    revised_text = regex.sub(r"\s+", " ", revised_text)
    return revised_text


def split_text(text, concat_bracket=False):
    split = text.split()
    new_split = []
    activation = False
    for word in split:
        if regex.match(r"^-(num|strange)\(", word):
            if regex.match(r".+\)-$", word):
                pass
            else:
                activation = True
            new_split.append(word)
        elif activation:
            if regex.match(r".+\)-$", word):
                activation = False
            new_split[-1] += " " + word
        else:
            new_split.append(word)

    if not concat_bracket:
        return new_split

    right_bracket = regex.compile(r"\)|\]|\}")
    left_bracket = regex.compile(r"\(|\[|\{")
    mini_batch = []
    batch = []
    activation = 0
    for word in new_split:
        if activation:
            mini_batch.append(word)
            rb = right_bracket.findall(word)
            lb = left_bracket.findall(word)
            if len(rb) - len(lb) == 1:
                activation = 0
                batch.append(" ".join(mini_batch))
                mini_batch.clear()
            elif activation:
                activation -= 1
            else:
                assert activation == 0
                batch += mini_batch
                mini_batch.clear()
        else:
            lb = left_bracket.findall(word)
            if not lb:
                batch.append(word)
            else:
                rb = right_bracket.findall(word)
                if len(lb) - len(rb) == 1 and word not in ['(', '[', '{']:
                    mini_batch.append(word)
                    activation = 3
                else:
                    batch.append(word)

    return batch


def identify_tag(label: str):
    if not label:
        return None
    elif regex.match(r"^-num([(].+[)])?-$", label):
        return "number"
    elif regex.match(r"^-strange\(.+\)-$", label):
        return 'strange'

    tag = regex.match(r"^-(?P<tag>c|u|n|a|e|i|s)\d\d\d\d\d\d-$", label)
    if not tag:
        return None
    tag_v = tag.group('tag')
    tag_dict = {'c': 'chemical', 'n': 'number', 'u': 'unit', 'e': 'element', 's': 'small_molecule', 'i': 'ion'}
    return tag_dict[tag_v]


def get_name(word, database):
    """ find original word
    :param word: (str) word
    :param database: (dict) dictionary of DataStorage, UnitStorage
    :return: (str) original word
    """

    ELEMENT = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K",
               "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
               "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I",
               "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
               "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr",
               "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf",
               "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og", "Uue"]

    tag = identify_tag(word)

    if not tag:
        return word
    elif tag == 'number':
        num = regex.match(r"^-num[(](?P<num>.+)[)]-$", word)
        if num:
            return num.group("num")
        return "-num-"
    elif tag == 'element':
        index = regex.match(r"^-e(\d+)-$", word).group(1)
        return ELEMENT[int(index)]
    elif tag == 'strange':
        strange = regex.match(r"^-strange[(](?P<strange>.+)[)]-$", word)
        if strange:
            return strange.group("strange")
        else:
            return None
    else:
        try:
            data_storage = database[tag]
        except KeyError:
            raise DatabaseError(f'{tag} not in database dictionary')

        return data_storage[word]


def _change_to_regex_string(chemical_list, original_list=None, return_as_str=False):
    """
    Change to regex form string / list
    :param chemical_list: (list) list of chemical
    :param original_list: (list) list that original change_to_re list
    :param return_as_str: (bool) if True, return (str). Else, return list
    :return: (list or str)
    """

    new_list = []
    if isinstance(chemical_list, str):
        chemical_list = [chemical_list]
    if not isinstance(original_list, list):
        original_list = []

    for chemical in chemical_list:
        if not chemical:
            continue
        elif not isinstance(chemical, str):
            chemical = str(chemical)

        chemical_revised = regex.sub(pattern=r"(?:\[|\]|\(|\)|\.|\,|\-|\*|\?|\{|\}|\$|\^|[|]|\+|\\)",
                                     string=chemical, repl=lambda t: r"\{}".format(t.group()))

        if chemical_revised in original_list or chemical_revised in new_list:
            continue
        elif regex.match(pattern=chemical_revised, string=chemical):  # Assert re_form == original form
            new_list.append(chemical_revised)
        else:
            raise AssertionError

    new_list += original_list
    new_list = sorted(new_list, reverse=True)

    if return_as_str:
        return r"|".join(new_list)
    else:
        return new_list
