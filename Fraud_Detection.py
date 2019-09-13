import pandas as pd 
data1=pd.read_csv('train_identity.csv')
data2= pd.read_csv('train_transaction.csv')

from sklearn import preprocessing
from category_encoders import *
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

def reduce_cols(row):
    return row.drop(columns=[col for col in row.columns if row[col].isnull().sum()>10000])
 

def roc_auc(row):
    columns=[]
    needed_cols=row.drop(columns=[col for col in row.columns if not row[col].dtype=='float64' if not row[col].dtype=='int64'])
    
    other_cols=list(set(row.columns)-set(needed_cols.columns))
    
    needed_cols=needed_cols.dropna()
    y = needed_cols['isFraud']
    X = pd.DataFrame()
    for cols in needed_cols.columns:
        
        X[cols] = needed_cols[cols]
                
        X_train, X_test, y_train, y_test = train_test_split(X[cols], y, test_size=0.33, random_state=47)
        clf = DecisionTreeClassifier(max_leaf_nodes=4)
        clf.fit(X_train.values.reshape(-1, 1), y_train)
        score=roc_auc_score(y_test, clf.predict_proba(X_test.values.reshape(-1, 1))[:, 1])
        if score>=0.54:
            columns.append(cols)
            
    return pd.concat([row['TransactionID'],row[columns],row[other_cols]],axis=1)


def encode(row):
    columns=['card4','ProductCD','card6']
    other_columns= list(set(row.columns)-set('isFraud'))
    enc=TargetEncoder(cols=columns, min_samples_leaf=20,smoothing=1.0).fit(row[other_columns],row['isFraud'])

    encoded_train=enc.transform(row[other_columns],row['isFraud']) 
    encoded_train['label']=row['isFraud']
    
    return encoded_train

    
def main(row):
    reduced= reduce_cols(row)
    
    roc= roc_auc(reduced)
    
    encoded= encode(roc)

    return encoded



