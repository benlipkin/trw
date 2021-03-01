import glob
import numpy as np
import pandas as pd
import pyconll as cnl


def get_corpus(lang):
    return {
        "English": ["EWT"],
        "Russian": ["SynTagRus"],
        "Chinese": ["PUD", "GSD"],
        "Arabic": ["PUD", "NYUAD", "PADT"],
        "Finnish": ["PUD", "TDT", "FTB"],
        "Turkish": ["PUD", "IMST", "GB", "BOUN"],
    }[lang]


def get_files(lang, corpus):
    return glob.glob(
        "../../../general/shared/UD_EWT/ud-treebanks-v2.4/UD_"
        + lang
        + "-"
        + corpus
        + "/*ud-*.conllu"
    )


def keep(token):
    if token.form in ["_", None]:
        return False
    if token.upos in ["PROPN", "PUNCT", "NUM", "SYM", "X", "None"]:
        return False
    for k, character in enumerate(token.form):
        if (k > 0) and (character.isupper()):
            return False
    return True


def extract_lex_freq(lang):
    lex_freq = {}
    for corpus in get_corpus(lang):
        for fname in get_files(lang, corpus):
            for sent in cnl.load_from_file(fname):
                for token in sent:
                    if keep(token):
                        word = token.form.lower()
                        if word in lex_freq:
                            lex_freq[word] += 1
                        else:
                            lex_freq[word] = 1
    return pd.DataFrame.from_dict(lex_freq, orient="index", columns=["count"])


def format_and_export(lex_freq, lang):
    fname = "../materials/lex_freq_" + lang.lower() + ".xlsx"
    lex_freq = lex_freq.rename_axis("token").reset_index()
    lex_freq["prob"] = lex_freq["count"] / lex_freq["count"].sum()
    lex_freq["logprob"] = np.log10(lex_freq["prob"])
    lex_freq.sort_values(by="token").to_excel(fname, index=False)


def main():
    for lang in ["English", "Russian", "Chinese", "Arabic", "Finnish", "Turkish"]:
        format_and_export(extract_lex_freq(lang), lang)


if __name__ == "__main__":
    main()
