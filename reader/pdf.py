from chemdataextractor import Document
from chemdataextractor.reader import PdfReader as CdeReader
from chemdataextractor.doc import Paragraph
from collections import defaultdict, Counter
import regex


from .reader_meta import Reader
from .error import ReaderError


class CDEPdfReader(Reader):
    suffix = '.pdf'
    
    @classmethod
    def parsing(cls, file):
        try:
            with open(file, 'rb') as f_cde:
                doc = Document.from_file(f_cde, readers=[CdeReader()])
            elements = [para for para in doc.elements if isinstance(para, Paragraph)]
        except Exception:
            raise ReaderError('ChemDataExtractor does not work. Please check your file.')

        if not elements:
            raise ReaderError('There are no paragraph in paper')
        return elements

    @classmethod
    def get_metadata(cls, file):
        metadata = defaultdict(type(None))
        elements = cls.parsing(file)

        month_dict = dict(january=1, february=2, march=3, april=4, may=5, june=6, july=7, august=8, september=9,
                          october=10, november=11, december=12, jan=1, feb=2, mar=3, apr=4, jun=6, jul=7, aug=8, sep=9,
                          oct=10, nov=11, dec=12)
        month_string = r"|".join(month_dict.keys())
        month_string = fr"\b({month_string})\b"

        year_counter = Counter()
        
        for element in elements:
            text = element.text
            if regex.search(r"(?i)\bdoi\b", text):
                doi_search = regex.search(r"\b\d\d\.\d\d\d\d/\S+", text)
                if doi_search:
                    metadata['doi'] = doi_search.group()

            elif regex.search(r"received|published|accepted", text, flags=regex.I) or \
                    regex.search(month_string, text, flags=regex.I):
                year_re = regex.findall(r"\b(20\d\d|19\d\d)\b", text)
                year_counter.update(year_re)

        if year_counter:
            year = year_counter.most_common(1)[0][0]
            metadata['date'] = year
        return metadata
