from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from matplotlib import pyplot
import pandas as pd

def get_dataset():
    X,y=make_classification(n_samples=1000,n_features=10,
                            n_informative=8,n_redundant=2,random_state=7)
    return X,y


def get_models():
    models=dict()
    n_trees=[10,50,100]
    arrange=[0.5,0.7]
    features=[3,5,7]
    depth=[3,5]
    for n in n_trees:
        for i in arrange:
            for z in features:
                for d in depth:
                    key=str("Ts:")+str(n)+str(" S:")+str(i)+str(" F:")+str(z)+str(" D:")+str(d)
                    models[key]=GradientBoostingClassifier(n_estimators=n,subsample=i,max_features=z,max_depth=d)
    return models




def evaluate_model(model,X,y):
    cv=RepeatedStratifiedKFold(n_splits=10,n_repeats=3,random_state=1)
    scores=cross_val_score(model,X,y,scoring='accuracy',cv=cv,n_jobs=-1)
    return scores
df=pd.read_csv('clean_dropped.csv')
X,y=df
models=get_models()
results,names=list(),list()
for name, model in models.items():
    scores=evaluate_model(model,X,y)
    results.append(scores)
    names.append(name)
    print('<%s%.3d(%.3f)'%(name,mean(scores),std(scores)))
pyplot.boxplot(results,labels=names,showmeans=True)
pyplot.show()
