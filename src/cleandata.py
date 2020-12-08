# cleandata.py
# Used for cleaning the data.
import numpy as np
import pandas as pd
from datamining import cleandata
from scipy.stats import spearmanr, chi2, chi2_contingency
from sklearn.decomposition import PCA
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

    #Clean sparse rows
    df = cleandata.remove_sparse_rows(df,0.4)

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
    df['checking_amt'] = pd.cut(x=df['checking_amt'],bins=pd.IntervalIndex.from_tuples([(-10000,-7500),(-7500,-5000),(-5000,-2500),(-2500, -1000),(-1000,-500),(-500,-100),(-100,-50),(-50,-20),(-20, -0.01),(-0.01, 0.01),(0.01, 20),(20,50),(50,100), (100, 500),(500,1000),(1000,2500),(2500,5000),(5000,7500),(7500,10000)]),labels=True)
    df['duration'] = pd.cut(x=df['duration'],bins=pd.IntervalIndex.from_tuples([(0, 12), (12, 24), (24, 36),(36,48),(48,60),(60,72)]),labels=True)
    df['credit_amount'] = pd.cut(x=df['credit_amount'],bins=pd.IntervalIndex.from_tuples([(200,500),(500,1000), (1000, 2500), (2500, 5000),(5000,7500),(7500,10000),(10000,12500),(12500,15000),(15000,20000)]),labels=True)
    df['savings'] = pd.cut(x=df['savings'],bins=pd.IntervalIndex.from_tuples([(0, 0.01), (0.01, 20),(20,50),(50,100), (100, 500),(500,1000),(1000,2500),(2500,5000),(5000,7500),(7500,10000)],closed="left"),labels=True)
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


def perform_and_display_spearman(df: pd.DataFrame, headers: List[str], threshold: float=0.85) -> None:
    with open('output/spearman.out','w') as f:
        f.write(f"Performing spearman tests on {headers}\n")
        for c1 in range(len(headers)):
            for c2 in range(c1,len(headers)):
                if c1 != c2:
                    df2 = df[[headers[c1],headers[c2]]].dropna()
                    X = df2[headers[c1]].values.reshape(-1,1)
                    Y = df2[headers[c2]].values.reshape(-1,1)
                    corr, p_value = spearmanr(X, Y)
                    del p_value
                    if abs(corr) > threshold:
                        f.write(f"RELATED ({headers[c1]}, {headers[c2]}): {corr}\n")
                    else:
                        f.write(f"NOT related ({headers[c1]}, {headers[c2]}):{corr}\n")
        f.write("Done")
    return


def perform_and_display_chi_square(df: pd.DataFrame, headers: List[str], significance: float) -> None:
    with open('output/chi2.out','w') as f:
        f.write(f"Performing Chi2 on {headers}\n")
        #Do it for every combination of categories
        for c1 in range(len(headers)):
            for c2 in range(c1+1,len(headers)):
                #Get those counts into the right shape
                df2 = df[[headers[c1],headers[c2]]].dropna()
                counts = df2.groupby([headers[c1],headers[c2]]).size()
                count_reformed = counts.unstack(headers[c1]).fillna(0)
                #Perform Chi2 Contingency
                chi, pval, dof, _ = chi2_contingency(count_reformed)
                critical_value = chi2.ppf(1-significance, dof-1)
                if chi > critical_value:
                    f.write(f"\n({headers[c1]}, {headers[c2]})\n")
                    f.write(f"[Independent: {not (chi > critical_value)}] (pval: {pval})")
                    f.write(f"Chi: {format(chi, '.2f')} cv: {format(critical_value, '.2f')}\n")
                    
        #print(pd.DataFrame(data=exp[:,:],columns=headers).round(2))
        f.write("\nDone")

    return


def perform_and_display_PCA(dataframe: pd.DataFrame, headers: List[str]) -> None:
    #Make copy
    df = dataframe.copy()

    #Perform PCA
    pca = PCA(n_components=len(headers))
    reduced_pca = pca.fit_transform(df[headers])

    for i in range (0,len(headers)):
        df['PC'+str(i+1)]=reduced_pca[:,i]
    
    #Print it
    PCs = []
    for i in range(0,len(pca.components_)):
        x = pca.components_[0,i]
        y = pca.components_[1,i]
        if x > 0 and y > 0:
            quad = 4
        elif x < 0 and y > 0:
            quad = 3
        elif x < 0 and y < 0:
            quad = 2
        else:
            quad = 1 
        PCs.append((quad, x,y,df.columns.values[i]))
    sortedPCs = sorted(PCs, key=lambda tup:(tup[0],tup[1]), reverse=True)
    with open("output/pca.out",'w') as f:
        f.write(f"Performing PCA on {headers}\n")
        for pc in sortedPCs:
            f.write(f"{pc}\n")
        f.write("Done")

    return


def main() -> None:
    #Input/output file locations
    IN_OUT_PAIRS = [("data/credit.csv","data/clean.csv"), ("data/clean.csv","data/clean_binned.csv"), ("data/clean.csv","data/clean_normalized.csv")]
    #Initial Cleans
    first_clean(infile=IN_OUT_PAIRS[0][0], outfile=IN_OUT_PAIRS[0][1])
    binned_clean(infile=IN_OUT_PAIRS[1][0], outfile=IN_OUT_PAIRS[1][1])
    normalized_clean(infile=IN_OUT_PAIRS[2][0], outfile=IN_OUT_PAIRS[2][1])

    #Build DF to run on spearman's
    numerical=['checking_amt','duration','credit_amount','savings','installment_commitment','residence_since','age','existing_credits','num_dependents']
    df=pd.read_csv(IN_OUT_PAIRS[2][1])
    for h in df.columns:
        if h not in numerical:
            df = df.drop([h], axis=1)
    perform_and_display_spearman(df=df,headers=numerical,threshold=0.75)
    
    #Build DF to run on chi's
    df=pd.read_csv(IN_OUT_PAIRS[1][1])
    #df = df.drop(numerical, axis=1)
    perform_and_display_chi_square(df,df.columns,significance=0.001)

    #Build DF for PCA
    df = pd.read_csv(IN_OUT_PAIRS[2][1])
    df = df[numerical].dropna()
    perform_and_display_PCA(df,numerical)


def drop_features():
    '''Drops features, determined by the previously done spearman,chi2, and PCA'''
    #Organize the dataframes to use
    df_list = [pd.read_csv("data/clean.csv"),pd.read_csv("data/clean_binned.csv"),pd.read_csv("data/clean_normalized.csv")]
    
    for df in df_list:
        #Drop location and application_date in favor of state (chi2)
        df.drop(["location","application_date","job"],axis=1,inplace=True)
        #Drop age,num_dependents, residence_since, existing_credits (PCA)
        df.drop(["age",'num_dependents', 'residence_since', 'existing_credits'],axis=1,inplace=True)
        df.drop(["duration",'installment_commitment'],axis=1,inplace=True)

    df_list[0].to_csv("data/clean_dropped.csv",index=False)
    df_list[1].to_csv("data/clean_dropped_binned.csv",index=False)
    df_list[2].to_csv("data/clean_dropped_normalized.csv",index=False)

if __name__ == "__main__":
    main()
    drop_features()
