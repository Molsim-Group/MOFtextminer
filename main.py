import argparse
from pathlib import Path

import tensorflow as tf


if __name__ == "__main__":
    print("hello")

    # Parse the arguments
    parser = argparse.ArgumentParser(description='Generating MOF dictionary')
    parser.add_argument(
        "-p",
        "--path",
        help="directory path containing papers (extend name : xml, html, pdf)",
    )

    parser.add_argument(
        "-f",
        "--filepath",
        type=str,
        help="one file path for a paper"
    )

    parser.add_argument(
        "-r",
        "--reader",
        type=str,
        default=None,
        help="HtmlReader, PdfReader, SpringerXmlReader, ElsevierXmlReader"
    )

    # args.reader
    args = parser.parse_args()

    # Check Reader and file extend
    if args.reader == "HtmlReader":
        extend = "html"
    elif args.reader == "PdfReader":
        extend = "pdf"
    elif args.reader in ["GeneralXmlReader", 'CDEXmlReader', 'ElsevierXmlReader']:
        extend = "xml"
    else:
        raise Exception("Please, choose proper Reader")
    print(f"Reader: {args.reader}, file extend : {extend}")

    # get filelist
    assert args.path
    filelist = list(Path(args.path).glob(f"*.{extend}"))

    # load MER model (tensorflow2.0 model)
    # bilstm_model = tf.keras.models.load_model("libs/mer/BILSTMCRF")
