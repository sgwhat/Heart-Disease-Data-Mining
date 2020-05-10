import numpy as np
import pandas as pd 

import lightgbm as lgb

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
# print(data.head())
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
data.drop(columns=['alco'], inplace=True)
data.info()
print(data.groupby("cardio").size())
data.dtypes
print(data.dtypes)
data.describe()
print(data.head(5))

#data.drop(columns=['id'], inplace=True)
#data.head()
data.info()
print(data.groupby("cardio").size())
data.dtypes
print(data.dtypes)

y = data['cardio']
X = data.drop(['cardio'], axis = 1)
print("Shape of X: {0}; positive example: {1}; negative: {2}".format(X.shape, y[y==1].shape[0], y[y==0].shape[0]))  # 查看数据的形状和类别分布
X_train, X_test, y_train, y_test = train_test_split(data.drop('cardio', 1), data['cardio'], test_size = .2, random_state=10) #split the data

#data.drop(columns=['alco','smoke','gender','active'], inplace=True)
params = {#'num_leaves': 60,
          #'min_data_in_leaf': 30,
          'objective': 'binary',
          'max_depth': 4,
	        'num_leaves':12,
          'learning_rate': 0.1,


#➜  Compile git:(master) ls
	  'feature_fraction':0.7,
          'min_child_samples':1,
	  'min_child_weight':8,
	  'bagging_fraction':1,
	  'bagging_freq':1,
	  'reg_alpha':0.005,
	  'reg_lambd':8,   
	#"min_sum_hessian_in_leaf": 6,
          "boosting": "gbdt",
          #"feature_fraction": 0.9,
          #"bagging_freq": 1,
          #"bagging_fraction": 0.8,
          #"bagging_seed": 11,
          #"lambda_l1": 0.1,
          # 'lambda_l2': 0.001,
          #"verbosity": -1,
          #"nthread": -1,
	  'cat_smooth':0,
	  'num_iterations':200,
          'metric': {'binary_logloss', 'auc'},
          #"random_state": 2019,
          # 'device': 'gpu'
          }

train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test)

model = lgb.train(params,train_data)
                    #num_round,
                    #valid_sets=[trn_data, val_data],
                    #verbose_eval=20,
                    #categorical_feature=cate_feature,
                    #early_stopping_rounds=60)


#模型预测
y_pred = model.predict(X_test)
#y_pred_quant=model.fit(X_train, y_train).predict_proba(X_test)[:, 1]
print("matchs: {0}/{1}".format(np.equal(y_pred.round(), y_test).shape[0], y_test.shape[0]))
xg_result = accuracy_score(y_test,y_pred.round())
print("Accuracy:", xg_result)


f1_score(y_test,y_pred.round())
print(classification_report(y_test,y_pred.round()))

confusion_matrix = confusion_matrix(y_test,y_pred.round())
print(confusion_matrix)
print('confusion_matrix:\n' , confusion_matrix)
#y_predict = model.predict(X_test)[:,1]

recall = confusion_matrix[1][1]/(confusion_matrix[1][1]+confusion_matrix[1][0])
precision = confusion_matrix[1][1]/(confusion_matrix[1][1]+confusion_matrix[0][1])
print("Recall:",recall,"Precision:",precision)

precisions,recalls,thresholds = precision_recall_curve(y_test,y_pred.round())
plt.plot(thresholds,precisions[:-1])
plt.plot(thresholds,recalls[:-1])
plt.grid()
#plt.show()

#Plot ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
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
plt.savefig('lightgbm_roc.png')
plt.show()
#Compute AUC
roc_auc = auc(fpr, tpr)
print('ROC_AUC_Score:', roc_auc)
print('successful')

parameters = {
	#'reg_alpha':[0.03,0.04,0.05,0.06,0.07]
	#'reg_lambda':range(1,10,1)
	'cat_smooth': [0]
}
'''
gbm = lgb.LGBMClassifier(objective = 'binary',
                         is_unbalance = True,
                         metric = 'binary_logloss,auc',
                         max_depth = 4,
                         num_leaves = 12,
                         learning_rate = 0.1,
                         feature_fraction = 0.7,
                         min_child_samples=32,
                         min_child_weight=32,
                         bagging_fraction = 1,
                         bagging_freq = 1,
                         reg_alpha = 0.005,
                         reg_lambda = 8,
                         cat_smooth = 0,
                         num_iterations = 200,   
                        )
'''
gbm = lgb.LGBMClassifier()
gsearch = GridSearchCV(gbm,param_grid=parameters, scoring='roc_auc', cv=3)#,param_grid=parameters)
gsearch.fit(X_train, y_train)
print('参数的最佳取值:{0}'.format(gsearch.best_params_))
print('最佳模型得分:{0}'.format(gsearch.best_score_))
print(gsearch.cv_results_['mean_test_score'])
print(gsearch.cv_results_['params'])

plt.figure(figsize=(12,6))
lgb.plot_importance(model, max_num_features=30)
plt.title("Featurertances")
plt.savefig('lightgbm.png')
plt.show()

