# cleandata.py
# Used for cleaning the data.
import numpy as np
import pandas as pd
from datamining import cleandata


def first_clean() -> None:
    """Initial cleaning, very basic"""
    INFILE = "credit.csv"
    OUTFILE = "output/clean2.csv"

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
    df["class"] = cleandata.typos(
        df["class"], ["bad", "good"], missing_values="?", ignore_missing=True
    )
    df["works_outside_US"] = cleandata.typos(
        df["works_outside_US"],
        ["no", "yes", "1"],
        missing_values="?",
        ignore_missing=True,
    )

    # Change absurd (likely erroneous) values in num_dependents to something within realm of possibility
    # [0, 130] (Can’t be younger than 0, ~10% higher than oldest recorded person)
    df['age'] = cleandata.bounding(df['age'],lower_bound=0,upper_bound=130, replace_with="?",missing_values="?",ignore_missing=True)
    # [0,1000] since nobody will have negative dependents and 1000 is ~10% higher than the highest number of recorded children had by a single person
    df["num_dependents"] = cleandata.bounding(df['num_dependents'],lower_bound=0,upper_bound=1000,replace_with="?",missing_values='?',ignore_missing=True)

    # Output from dataframe into outfile
    df.to_csv(OUTFILE, index=False)

    return


def binned_clean() -> None:
    '''clean.csv but with binned values'''
    INFILE = "output/clean.csv"
    OUTFILE = "output/clean_binned.csv"

    # Build Dataframe
    df: pd.DataFrame = pd.read_csv(INFILE)

    # Bin all of the numerical data


    # Output from dataframe into outfile
    df.to_csv(OUTFILE)

    return


def normalized_clean() -> None:
    '''clean.csv but with normalized values'''
    INFILE = "output/clean.csv"
    OUTFILE = "output/clean_normalized.csv"

    # Build Dataframe
    df: pd.DataFrame = pd.read_csv(INFILE)

    # Normalize all of the numerical data
    for h in ['checking_amt','duration','credit_amount','savings','installment_commitment','residence_since','age','existing_credits','num_dependents']:
        df[h] = cleandata.normalize(df[h])

    # Output from dataframe into outfile
    df.to_csv(OUTFILE)

    return 


if __name__ == "__main__":
    first_clean()
    binned_clean()
    normalized_clean()
