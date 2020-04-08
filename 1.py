from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('bank-additional-full.csv',sep=";")

numberVar=['age','duration','campaign','pdays','previous','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed']                           #10个数值变量
categoryVar = ['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome','y']

# 2.1 缺失值检查
total = df.shape[0]
cols = categoryVar +['pdays']
for col in cols:
    v = df[col].value_counts().to_dict()
    if 'unknown' in v.keys():
        unCount = v['unknown']
    elif 'nonexistent' in v.keys():
        unCount = v['nonexistent']
    elif 999 in v.keys():
        unCount = v[999]
    else:
        continue
    print("%-10s: %5.1f%%"%(col,unCount/total*100))

# 2.2 高缺失比例的变量处理
plt.hist(df['pdays'])
plt.show()

dfPdays = df.loc[df.pdays != 999, 'pdays']
plt.hist(dfPdays)
plt.show()
print(df.pdays.value_counts())
pdaysDf = df['pdays'].apply(lambda x : int(x/5)*5)
print(pd.crosstab(pdaysDf,df['poutcome']))

# 2.3 default（信用违约）缺失值分析和处理
print(df.default.value_counts())
def defaultAsso(dataset, col):
    tab = pd.crosstab(dataset['default'],dataset[col]).apply(lambda x: x/x.sum() *100)
    tab_pct = tab.transpose()
    x = tab_pct.index.values
    plt.figure(figsize=(14,3))
    plt.plot(x, tab_pct['unknown'],color='green',label='unknown')
    plt.plot(x, tab_pct['yes'],color='blue',label='yes')
    plt.plot(x, tab_pct['no'],color='red',label='no')
    plt.legend()
    plt.xlabel(col)
    plt.ylabel('rate')
    plt.show()

defaultAsso(df, 'job')
defaultAsso(df, 'education')
defaultAsso(df, 'marital')

df['ageGroup'] = df['age'].apply(lambda x: int(x/5)*5)
print(df.ageGroup.value_counts())
defaultAsso(df, 'ageGroup')
df.drop("ageGroup", inplace=True, axis=1)

df['default'] = df["default"].map({'no':0,'yes':1,'unknown':1})
print(df.default.value_counts())

# 2.4 处理极少量缺失比例的变量
df.drop(df[df.job == 'unknown'].index, inplace=True,axis=0)
print(df.job.value_counts())
df.drop(df[df.marital == 'unknown'].index, inplace=True,axis=0)
print(df.marital.value_counts())

print(pd.crosstab(df['housing'],df['loan']))
df.drop(df[df.housing == 'unknown'].index, inplace=True,axis=0)
print(df.housing.value_counts())
print(df.loan.value_counts())

# 3. 将分类变量转为数值
### 3.1 只有两种取值的变量
df['y'] = df["y"].map({'no':0,'yes':1})
print(df.y.value_counts())
df['contact'] = df['contact'].map({'cellular':0,'telephone':1})
print(df.contact.value_counts())
df['housing'] = df["housing"].map({'no':0,'yes':1})
print(df.housing.value_counts())
df['loan'] = df["loan"].map({'no':0,'yes':1})
print(df.loan.value_counts())
print(df[['y','default','contact','housing','loan']].head())

# 3.2 有序分类变量编码
values = ["unknown","illiterate", "basic.4y", "basic.6y", "basic.9y", "high.school",  "professional.course", "university.degree"]
levels = range(0,len(values))
dict_levels = dict(zip(values, levels))
for v in values:
    df.loc[df['education'] == v, 'education'] = dict_levels[v]
print(df.education.value_counts())
df['education'] = df['education'].astype(int)

# 3.2.3.将无序分类变量转为虚拟变量
df = pd.get_dummies(df, columns = ['job'])
print(df.info())
df = pd.get_dummies(df, columns = ['marital'])
print(df.info())
df = pd.get_dummies(df, columns = ['poutcome'])
print(df.info())
df = pd.get_dummies(df, columns = ['month'])
print(df.info())
df = pd.get_dummies(df, columns = ['day_of_week'])
print(df.info())

# 3.3.基于机器学习的缺失值补充
def train_predict_unknown(trainX, trainY, testX):
    forest = RandomForestClassifier(n_estimators=100)
    forest = forest.fit(trainX, trainY)
    test_predictY = forest.predict(testX).astype(int)
    return pd.DataFrame(test_predictY,index=testX.index)
test_data = df[df['education'] == 0]
testX = test_data.drop('education', axis=1)
train_data = df[df['education'] != 0]
trainY = train_data['education']
trainX = train_data.drop('education', axis=1)

test_data['education'] = train_predict_unknown(trainX, trainY, testX)
print(test_data.education.value_counts())

# 将测试集与训练集合并成一张表格：
df = pd.concat([train_data, test_data])
print(df.shape)
print(df.education.value_counts())

# 3.1.数值变量标准化
def scaleColumns(data, cols_to_scale):
    scaler = StandardScaler()
    idx = data.index.values
    for col in cols_to_scale:
        x = scaler.fit_transform(pd.DataFrame(data[col]))
        data[col] = pd.DataFrame(x, columns=['col'], index=idx)
    return data

df = scaleColumns(df, numberVar+['education'])
print(df.head())

# 3.4.特征选择
df.drop("education", inplace=True, axis=1)
print(df.shape)
print(df.info())

# 3.2.保存预处理数据
df = shuffle(df)
print(df.shape)
df.to_csv('bank-preprocess.csv',index=False)


