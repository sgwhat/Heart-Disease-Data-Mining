import numpy as np
import pandas as pd 

from xgboost import XGBClassifier
import xgboost as xgb

from sklearn.metrics import auc
from sklearn.metrics import roc_curve 
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import export_graphviz
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV


warnings.filterwarnings('ignore')

df=pd.read_csv('/usr/local/dataset/cardio_train.csv', 
                        sep = ';' 
                        #names = ['age','gender','height','weight',
                                 #'ap_hi','ap_lo','cholesterol','glu',
                                 #'smoke','alco','active',
                                 #'cardio']
                                 )

data = df[(df['ap_lo'] <= df['ap_hi']) &
                 (df['height'] >= df['height'].quantile(0.025)) &
                 (df['height'] <= df['height'].quantile(0.975)) &
                 (df['weight'] >= df['weight'].quantile(0.025)) &
                 (df['weight'] <= df['weight'].quantile(0.975))]
1 - data.shape[0] / df.shape[0]
print(data.groupby("cardio").size())
data.drop(columns=['id'], inplace=True)
data['age']=round(data['age']/365).apply(lambda x: int(x))

print(f"In {data[data['ap_hi'] < data['ap_lo']].shape[0]} obeservation ap_hi is lower than ap_low, which is incorrect.")
print('_'*80)
print()
print("Let's remove them:")

data = data[data['ap_hi'] > data['ap_lo']].reset_index(drop=True)
data.head()

def BMI(data):
    return data['weight'] / (data['height']/100)**2
data['bmi'] = data.apply(BMI, axis=1)
data.drop(columns=['height','weight'], inplace=True)
# print(data.head())
#data.drop(columns=['id'], inplace=True)
#data.head()
data.info()
print(data.groupby("cardio").size())
data.dtypes
print(data.dtypes)

y = data['cardio']
x = data.drop(['cardio'], axis = 1)
print("Shape of X: {0}; positive example: {1}; negative: {2}".format(x.shape, y[y==1].shape[0], y[y==0].shape[0]))  # 查看数据的形状和类别分布
x_train, x_test, y_train, y_test = train_test_split(data.drop('cardio', 1), data['cardio'], test_size = .2, random_state=10) #split the data

#model=XGBClassifier()
model = XGBClassifier(learning_rate=0.055,n_estimators=400,max_depth=5,
			min_child_weight=32,gamma=0.1,subsample=0.7,
			colsample_bytree=1,reg_alpha=0.04,reg_lambda=1,
			scale_pos_weight=1,eta=0.1
			)

model.fit(x_train, y_train)
train_score = model.score(x_train, y_train)
test_score = model.score(x_test, y_test)
print("train score: {train_score:.6f}; test score: {test_score:.6f}".format(train_score=train_score, test_score=test_score))

#模型预测
y_pred = model.predict(x_test)
print("matchs: {0}/{1}".format(np.equal(y_pred, y_test).shape[0], y_test.shape[0]))
xg_result = accuracy_score(y_test,y_pred)
print("Accuracy:", xg_result)

f1_score(y_test,y_pred)
print(classification_report(y_test,y_pred))

confusion_matrix = confusion_matrix(y_test,y_pred)
print(confusion_matrix)
print('confusion_matrix:\n' , confusion_matrix)
y_predict = model.predict_proba(x_test)[:,1]
y_pred_quant=model.fit(x_train, y_train).predict_proba(x_test)[:, 1]

recall = confusion_matrix[1][1]/(confusion_matrix[1][1]+confusion_matrix[1][0])
precision = confusion_matrix[1][1]/(confusion_matrix[1][1]+confusion_matrix[0][1])
print("Recall:",recall,"Precision:",precision)

precisions,recalls,thresholds = precision_recall_curve(y_test,y_pred)
plt.plot(thresholds,precisions[:-1])
plt.plot(thresholds,recalls[:-1])
plt.grid()
plt.show()

#Plot ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_quant)
fig, ax = plt.subplots()
ax.plot(fpr, tpr)
ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC curve for diabetes classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
plt.show()
plt.savefig('xgb_roc_2.png')
#Compute AUC
roc_auc = auc(fpr, tpr)
print('ROC_AUC_Score:', roc_auc)
print('successful')

plt.figure(figsize=(12,6))
xgb.plot_importance(model, max_num_features=30)
plt.title("Featurertances")
plt.savefig('xgboost.png')
plt.show()
'''
param_test1 = {'n_estimators':[45,49,50,51,55]}
gsearch1 = GridSearchCV(estimator = XGBClassifier(learning_rate=0.1, n_estimators=500, max_depth= 5, min_child_weight= 1, seed= 0,subsample=0.8, colsample_bytree=0.8, gamma=0, reg_alpha=0, reg_lambda=1), 
                       param_grid = param_test1, scoring='roc_auc',cv=5)
gsearch1.fit(x_train,y_train)
print( gsearch1.best_params_, gsearch1.best_score_)
'''



