import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import ttest_ind as ttest


def memoize(function):
    memo = {}

    def wrapper(*args):
        if args in memo:
            return memo[args]
        else:
            rv = function(*args)
            memo[args] = rv
            return rv

    return wrapper


def load_data(lang, seq):
    return pd.read_excel(
        "../strings/%s/%dwords_17sents_first5start_TEST_3X.xlsx" % (lang, seq),
        engine="openpyxl",
    )


@memoize
def load_lex_freq(lang):
    return pd.read_excel(
        "../materials/lex_freq_%s.xlsx" % (lang.lower()), engine="openpyxl"
    )


def get_first_idx(data):
    return data["start_ind"].values


def get_last_idx(data, seq):
    return get_first_idx(data) + seq


def get_tags(type):
    if type == "content":
        return ["ADJ", "ADV", "NOUN", "VERB"]
    elif type == "function":
        return ["ADP", "AUX", "CCONJ", "DET", "INTJ", "PART", "PRON", "SCONJ"]
    else:
        raise LookupError()


def get_type_count(pos, type):
    tags = get_tags(type)
    count = np.empty(pos.size)
    for idx, taglist in enumerate(pos):
        count[idx] = np.sum([tag in tags for tag in taglist])
    return count


def get_cf_ratio(data, seq):
    pos = data["pos"].str.split().str[:-2].values
    content = get_type_count(pos, "content")
    function = get_type_count(pos, "function")
    return (content - function) / (content + function)


def token_freq(token, lookup):
    return lookup[lookup.token == token].logprob.values[0]


def token_exists(token, lookup):
    return token in lookup.token.values


def get_mean_freq(tokens, lookup):
    return np.mean(
        [
            token_freq(token, lookup)
            if token_exists(token, lookup)
            else lookup.logprob.min()
            for token in tokens
        ]
    )


def get_tokens(data):
    return data.iloc[:, 6:]


def get_lex_freq(data, lang):
    lookup = load_lex_freq(lang)
    tokens = get_tokens(data)
    lex_freq = np.empty(tokens.shape[0])
    for idx in range(lex_freq.size):
        lex_freq[idx] = get_mean_freq(tokens.iloc[idx, :].values, lookup)
    return lex_freq


def get_word_len(data):
    tokens = get_tokens(data)
    word_len = np.empty(tokens.shape[0])
    for idx in range(word_len.size):
        word_len[idx] = np.mean([len(token) for token in tokens.iloc[idx, :].values])
    return word_len


def get_features(lang, seq):
    data = load_data(lang, seq)
    return (
        pd.DataFrame(
            {
                "First Index": get_first_idx(data),
                # "Last Index": get_last_idx(data, seq),
                "Content Function Ratio": get_cf_ratio(data, seq),
                "Lexical Frequency": get_lex_freq(data, lang),
                "Word Length": get_word_len(data),
            }
        ),
        data,
    )


def initialize_indices(features):
    indices = {}
    for seq in features:
        indices[seq] = np.ones(features[seq].shape[0], dtype="bool")
    return indices


def check_pairwise_stats(features, indices):
    seqs = sorted(features.keys())
    pairs, feats, tstats, pvals = [], [], [], []
    for pair in combinations(seqs, 2):
        for feature in features[seqs[0]].columns:
            [tstat, pval] = ttest(
                features[pair[0]].loc[indices[pair[0]], feature],
                features[pair[1]].loc[indices[pair[1]], feature],
            )
            pairs.append(pair)
            feats.append(feature)
            tstats.append(tstat)
            pvals.append(pval)
    return pd.DataFrame(
        {"pair": pairs, "feature": feats, "t-stat": tstats, "p-value": pvals}
    )


def is_valid(stats):
    return stats["p-value"].min() >= 0.05


def get_samples(features, indices, sample, group):
    return features[sample.pair[group]].loc[indices[sample.pair[group]], sample.feature]


def weight(indices, sample, group):
    return indices[sample.pair[group]].sum() / indices[sample.pair[group]].size


def set_indices(indices, sample, sx, sm, group):
    indices[sample.pair[group]][np.abs(sx - sm).idxmax()] = 0


def trim_worst_outlier(features, indices, stats):
    sample = stats.iloc[stats["p-value"].idxmin()]
    s0 = get_samples(features, indices, sample, 0)
    s1 = get_samples(features, indices, sample, 1)
    sm = np.mean(pd.concat((s0, s1)).values)
    if weight(indices, sample, 0) >= weight(indices, sample, 1):
        set_indices(indices, sample, s0, sm, 0)
    else:
        set_indices(indices, sample, s1, sm, 1)
    return indices


def get_indices_to_keep(features, lang):
    indices = initialize_indices(features)
    stats = check_pairwise_stats(features, indices)
    export_features(features, indices, lang, "Initial")
    export_stats(stats, lang, "Initial")
    while not is_valid(stats):
        indices = trim_worst_outlier(features, indices, stats)
        stats = check_pairwise_stats(features, indices)
    export_features(features, indices, lang, "Final")
    export_stats(stats, lang, "Final")
    return indices


def export_features(features, indices, lang, phase):
    print("Exporting " + phase + " " + lang + " Features...")
    for seq in features:
        features[seq][indices[seq]].to_csv(
            "../stats/%s_%s_%dwords_features.csv" % (phase.lower(), lang.lower(), seq),
            index=False,
        )


def export_stats(stats, lang, phase):
    print("Exporting " + phase + " " + lang + " Pairwise Stats...")
    stats.to_csv(
        "../stats/%s_%s_pairwise_stats.csv" % (phase.lower(), lang.lower()), index=False
    )


def export_strings(strings, indices, lang):
    print("Exporting Final " + lang + " Strings...")
    for seq in strings:
        strings[seq][indices[seq]].to_excel(
            "../matched/%s_%dwords.xlsx" % (lang.lower(), seq), engine="openpyxl"
        )
    print_status(strings, indices)


def print_status(strings, indices):
    print("\nSEQLEN\tINITIAL\tFINAL")
    for seq in strings:
        print(
            "%d\t%d\t%d"
            % (seq, (strings[seq]).shape[0], (strings[seq][indices[seq]]).shape[0])
        )


def main():
    print("\nRunning Main...")
    for lang in ["Arabic", "Chinese", "English", "Finnish", "Russian", "Turkish"]:
        print("\nProcessing " + lang + "...")
        features, strings = {}, {}
        for seq in [2, 3, 4, 5, 6, 8, 10, 12]:
            features[seq], strings[seq] = get_features(lang, seq)
        indices = get_indices_to_keep(features, lang)
        export_strings(strings, indices, lang)
    print("\nDone.\n")


if __name__ == "__main__":
    main()
