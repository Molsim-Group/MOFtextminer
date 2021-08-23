from bs4 import BeautifulSoup
from collections import defaultdict
from chemdataextractor import Document
from chemdataextractor.doc import Paragraph

from .reader_meta import Reader
from .error import ReaderError


class CDEHtmlReader(Reader):
    suffix = '.html'
        
    @classmethod
    def parsing(cls, file):
        try:
            with open(file, 'rb') as f_cde:
                doc = Document.from_file(f_cde)
            elements = [para for para in doc.elements if isinstance(para, Paragraph)]
        except Exception:
            raise ReaderError('ChemDataExtractor does not work. Please check your file.')

        if not elements:
            raise ReaderError('There are no paragraph in paper')
        return elements

    @classmethod
    def get_metadata(cls, file):
        with open(file, encoding='UTF-8') as f:
            bs = BeautifulSoup(f, 'html.parser')

        metadata = defaultdict(type(None))
        metadata['author_list'] = []
        for meta in bs.find_all('meta'):
            if not meta.get('name'):
                continue

            name = meta.get('name')
            if (name.find('citation_') != -1 and name != 'citation_reference') or name.find('dc.') != -1:
                tag = meta.get('name').replace('citation_', '').replace('dc.', '')
                text = meta.get('content')
                if tag == 'doi' or tag == 'Identifier':
                    metadata['doi'] = text
                elif tag == 'title' or tag == 'Title':
                    metadata['title'] = text
                elif tag == 'journal_title':
                    metadata['journal'] = text
                elif tag in ['publication_date', 'Date', 'date']:
                    metadata['date'] = text
                elif tag == 'author' or tag == 'Creator':
                    metadata['author_list'].append(text, )

        return metadata
