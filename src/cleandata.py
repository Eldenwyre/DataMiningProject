# cleandata.py
# Used for cleaning the data.
import numpy as np
import pandas as pd
from datamining import cleandata
from scipy.stats import spearmanr
from typing import List
def first_clean(infile: str, outfile: str) -> None:
    """Initial cleaning, very basic"""
    INFILE = infile
    OUTFILE = outfile

    # Build Dataframe
    df: pd.DataFrame = pd.read_csv(INFILE)

    # Drop the verified col (too sparse to use, nearly no value can be gained)
    df = df.drop(["verified"], axis=1)

    # Replace any nan values with '?' for some uniformity
    cleandata.fix_nan(df, replace_with="?")

    # Get dates in consistent format, dropping day to get larger grouping of dates.
    df["application_date"] = cleandata.dates(
        df["application_date"], f="%m/%Y", missing_values="?", ignore_missing=True
    )

    # Uniformize foreign_worker, class, works_outside_US
    df["foreign_worker"] = cleandata.typos(
        df["foreign_worker"],
        ["no", "yes", "1"],
        missing_values="?",
        ignore_missing=True,
    )
    df["foreign_worker"].replace(to_replace='1',value='yes',inplace=True,)
    df["class"] = cleandata.typos(
        df["class"], ["bad", "good"], missing_values="?", ignore_missing=True
    )
    df["works_outside_US"] = cleandata.typos(
        df["works_outside_US"],
        ["no", "yes", "1"],
        missing_values="?",
        ignore_missing=True,
    )
    df["works_outside_US"].replace(to_replace='1',value='yes',inplace=True,)

    # Change absurd (likely erroneous) values in num_dependents to something within realm of possibility
    # [0, 130] (Can’t be younger than 0, ~10% higher than oldest recorded person)
    df['age'] = cleandata.bounding(df['age'],lower_bound=0,upper_bound=130, replace_with="?",missing_values="?",ignore_missing=True)
    # [0,1000] since nobody will have negative dependents and 1000 is ~10% higher than the highest number of recorded children had by a single person
    df["num_dependents"] = cleandata.bounding(df['num_dependents'],lower_bound=0,upper_bound=1000,replace_with="?",missing_values='?',ignore_missing=True)

    for h in df.columns:
        df[h] = np.where(df[h] == "?",np.nan,df[h])

    # Output from dataframe into outfile
    df.to_csv(OUTFILE, index=False)

    return


def binned_clean(infile: str, outfile: str) -> None:
    '''clean.csv but with binned values'''
    INFILE = infile
    OUTFILE = outfile

    # Build Dataframe
    df: pd.DataFrame = pd.read_csv(INFILE)

    # Bin all of the numerical data
    df['checking_amt'] = pd.cut(x=df['checking_amt'],bins=pd.IntervalIndex.from_tuples([(-10000, -1000), (-1000, -0.01), (-0.01, 0.01),(0.01,1000),(1000,10000)]),labels=True)
    df['duration'] = pd.cut(x=df['duration'],bins=pd.IntervalIndex.from_tuples([(0, 12), (12, 24), (24, 36),(36,48),(48,60),(60,72)]),labels=True)
    df['credit_amount'] = pd.cut(x=df['credit_amount'],bins=pd.IntervalIndex.from_tuples([(200,500),(500,1000), (1000, 2500), (2500, 5000),(5000,7500),(7500,10000),(10000,12500),(12500,15000),(15000,20000)]),labels=True)
    df['savings'] = pd.cut(x=df['savings'],bins=pd.IntervalIndex.from_tuples([(0, 0.01), (0.01, 100), (100, 500),(500,1000),(1000,2500),(2500,5000),(5000,7500),(7500,10000)],closed="left"),labels=True)
    df['age'] = pd.cut(x=df['age'],bins=pd.IntervalIndex.from_tuples([(0, 10), (10, 20), (20, 30),(30,40),(40,50),(50,60),(60,70),(70,80),(80,90),(100,140)]),labels=True)

    # Output from dataframe into outfile
    df.to_csv(OUTFILE,index=False)

    return


def normalized_clean(infile: str, outfile: str) -> None:
    '''clean.csv but with normalized values'''
    INFILE = infile
    OUTFILE = outfile

    # Build Dataframe
    df: pd.DataFrame = pd.read_csv(INFILE)

    # Normalize all of the numerical data
    for h in ['checking_amt','duration','credit_amount','savings','installment_commitment','residence_since','age','existing_credits','num_dependents']:
        df[h] = cleandata.normalize(df[h])

    # Output from dataframe into outfile
    df.to_csv(OUTFILE,index=False)

    return 


def perform_and_display_spearman(df: pd.DataFrame, headers: List[str], outfile: str) -> None:
    print(f"Performing spearman tests on {headers}")
    for c1 in range(len(headers)):
        X = df[headers[c1]].values.reshape(-1,1)
        for c2 in range(c1,len(headers)):
            if c1 != c2:
                Y = df[headers[c2]].values.reshape(-1,1)
                corr, p_value = spearmanr(X, Y)
                print(f"({headers[c1]}, {headers[c2]}) : {abs(corr)}")
    print("Done")
    return



if __name__ == "__main__":
    IN_OUT_PAIRS = [("credit.csv","output/clean.csv"), ("output/clean.csv","output/clean_binned.csv"), ("output/clean.csv","output/clean_normalized.csv")]
    first_clean(infile=IN_OUT_PAIRS[0][0], outfile=IN_OUT_PAIRS[0][1])
    binned_clean(infile=IN_OUT_PAIRS[1][0], outfile=IN_OUT_PAIRS[1][1])
    normalized_clean(infile=IN_OUT_PAIRS[2][0], outfile=IN_OUT_PAIRS[2][1])
    #perform_and_display_spearman()
