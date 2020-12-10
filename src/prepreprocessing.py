# prepreprocessing.py
import pandas as pd
import numpy as np
from datamining import prepreprocessing as ppp
import sys


def print_unique_col_info(df: pd.DataFrame) -> None:
    for col in df.columns:
        vals = ppp.unique_values(df[col])
        count = ppp.count_complete_col(df[col], "?")
        print(f"\n{col} has {len(vals)} unique values")
        print(f"It has {count} nonmissing entries. ({count/len(df)*100}%)")
        # Since states are included, using 50 as cutoff
        # Print any potential nominal attributes
        if len(vals) <= 51:
            print(f"   These values are: {vals}")
        else:
            strings = []
            nums = []
            for val in vals:
                try:
                    float(val)
                    nums.append(float(val))
                except:
                    strings.append(val)
            nums = sorted(nums)
            if len(nums) > len(strings):
                print(
                    f"Suspected numerical data. Lowest 5 numbers {nums[0:5]}. Highest 5 numbers {nums[-5:]}"
                )
                print(
                    f"It also contains the following strings (possibly erroneous?): {strings}"
                )
            else:
                print(f"Nominal Data suspected, but with high counts.")

    return


def print_row_info(df: pd.DataFrame) -> None:
    complete_rows = ppp.complete_count(df, "?")
    row_count = len(df)
    print(f"There are {complete_rows} rows without missing values.")
    print(f"This means there is a ratio of {complete_rows/row_count} complete rows.\n")

    return


def main(file: str) -> None:
    df = pd.read_csv(file)
    print_row_info(df)
    print_unique_col_info(df)


if __name__ == "__main__":
    main(sys.argv[1])
