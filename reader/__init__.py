from .pdf import CDEPdfReader
from .html import CDEHtmlReader
from .xml import ElsevierXmlReader, GeneralXmlReader, CDEXmlReader
from .error import ReaderError

Default_readers = {'.html': CDEHtmlReader, '.pdf': CDEPdfReader, '.xml': CDEXmlReader}
