from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import numpy as np
import pandas as pd

def loadDataSet():
    return [['m','o','n','k','e','y'],['d','o','n','k','e','y']
            ,['m','a','k','e'],['m','u','c','k','y']
            ,['c','o','o','k','i','e']]

datanew = np.array(loadDataSet())
oht = TransactionEncoder()
oht_ary = oht.fit(datanew).transform(datanew)
df = pd.DataFrame(oht_ary, columns=oht.columns_)
# print(df.head())
df_dfe = apriori(df, min_support=0.6, use_colnames=True)
# print(df_dfe)

rule = association_rules(df_dfe, metric="confidence", min_threshold=0.8)
print(rule)