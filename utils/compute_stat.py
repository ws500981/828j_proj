import pandas as pd
import numpy as np
import os

def unique_words(df):
    a = ""
    for i in df["text"]:
        a += i
    return(len(list(set(a.split()))))

def yes_or_no(df):
    n_yes = df[df["label"]=="1"].shape[0]
    n_no = df[df["label"]=="0"].shape[0]
    return n_yes, n_no

if __name__ == "__main__":
    df_real = pd.read_csv("./../data/fin_data/real_lort.csv", dtype=object)
    df_syn_vars = pd.read_csv("./../data/fin_data/syn_lort_vars.csv", dtype=object)
    df_syn_nums = pd.read_csv("./../data/fin_data/syn_lort_nums.csv", dtype=object)
    df_syn_fixed = pd.read_csv("./../data/fin_data/syn_lort_fixed.csv", dtype=object)
    df_test = pd.read_csv("./../data/fin_data/test_lort.csv", dtype=object)

    print("data", "unique words", "yes", "no")
    print("Real data", unique_words(df_real), yes_or_no(df_real))
    print("Syn var data", unique_words(df_syn_vars), yes_or_no(df_syn_vars))
    print("Syn numbers data", unique_words(df_syn_nums), yes_or_no(df_syn_nums))
    print("Syn fixed data", unique_words(df_syn_fixed), yes_or_no(df_syn_fixed))
    print("test data", unique_words(df_test), yes_or_no(df_test))


