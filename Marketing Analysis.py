#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')


# In[2]:


data=pd.read_csv('train.csv',sep=';')


# In[3]:


data


# marital=le,
# job=le or ohe
# education=le,
# default,housing,loan=mapping,
# contact=le,
# poutcome=le
# 
# 
# 

# In[4]:


data['pdays'].value_counts()


# In[5]:


data.info()


# In[6]:


data.describe().T


# # Removing unnecessay features

# In[7]:


data.drop(['day','month','previous'],axis=1,inplace=True) 


# In[8]:


data.rename(columns={'y':'Bank_Deposit'},inplace=True)


# #  Correlation between the numeric columns

# In[9]:


correlation=data.corr()
heatmap=sns.heatmap(correlation,annot=True)
heatmap.set (title = "Correlation matrix of dataset\n")  
plt. show ()  


# # Poor correlation between the numeric features.  Hence none of them is ignorable.

# In[10]:


ajg=data.groupby(['age','job'])['Bank_Deposit'].count().reset_index()
ajg_1 = ajg.pivot(index='job', columns='age')['Bank_Deposit'].fillna(0)


# In[11]:


ajgh = px.imshow(ajg_1, labels=dict(x="age", y="job", color="Bank_Deposit"), x=ajg_1.columns, y=ajg_1.index,
                    aspect="auto", title='Age_job_bank deposit analysis',text_auto=True)


# In[12]:


ajgh


# In[13]:


data.columns


# In[13]:


ahb=data.groupby(['age','housing'])['Bank_Deposit'].count().reset_index()
ahb_1 = ahb.pivot(index='housing', columns='age')['Bank_Deposit'].fillna(0)


# In[14]:


ahbh = px.imshow(ahb_1, labels=dict(x="age", y="housing", color="Bank_Deposit"), x=ahb_1.columns, y=ahb_1.index,
                    aspect="auto", title='Age_housing_loan_bank deposit analysis')


# In[15]:


ahbh


# In[16]:


apb=data.groupby(['age','loan'])['Bank_Deposit'].count().reset_index()
apb_1 = apb.pivot(index='loan', columns='age')['Bank_Deposit'].fillna(0)


# In[17]:


apbh = px.imshow(apb_1, labels=dict(x="age", y="loan", color="Bank_Deposit"), x=apb_1.columns, y=apb_1.index,
                    aspect="auto", title='Age_personal_loan_bank deposit analysis')


# In[18]:


apbh


# In[19]:


ab=data.groupby('age')['Bank_Deposit'].count().sort_values(ascending=False).reset_index()


# In[20]:


ab


# In[21]:


px.bar(ab,x='age',y='Bank_Deposit',color='Bank_Deposit',width=2000)


# In[22]:


hb=data.groupby('housing')['Bank_Deposit'].count().reset_index()


# In[23]:


px.bar(hb,x='housing',y='Bank_Deposit',color='Bank_Deposit',text='Bank_Deposit')


# In[24]:


pb=data.groupby('loan')['Bank_Deposit'].count().reset_index()
px.bar(pb,x='loan',y='Bank_Deposit',color='Bank_Deposit',text='Bank_Deposit')


# marital=le, job=le or ohe education=le, default,housing,loan=mapping, contact=le, poutcome=le

#                       Preprocessing

# In[ ]:





# In[121]:


from sklearn import preprocessing 


# In[122]:


label_encoder = preprocessing.LabelEncoder() 


# In[123]:


data['marital']=label_encoder.fit_transform(data['marital']) 
data['job']=label_encoder.fit_transform(data['job']) 
data['education']=label_encoder.fit_transform(data['education']) 
data['contact']=label_encoder.fit_transform(data['contact']) 
data['poutcome']=label_encoder.fit_transform(data['poutcome']) 
data['default']=data['default'].map({'yes':1,'no':0})
data['housing']=data['housing'].map({'yes':1,'no':0})
data['loan']=data['loan'].map({'yes':1,'no':0})
   


# In[124]:


data


# In[125]:


GroupedData=data.groupby(by= 'Bank_Deposit').size()
GroupedData.plot.bar()


# In[126]:


data_test=pd.read_csv('test.csv',sep=';')


# In[127]:


data_test


# In[131]:


data_test.drop(['day','month','previous'],axis=1,inplace=True) 
data_test.rename(columns={'y':'Bank_Deposit'},inplace=True)


# In[132]:


data_test['marital']=label_encoder.fit_transform(data_test['marital']) 
data_test['job']=label_encoder.fit_transform(data_test['job']) 
data_test['education']=label_encoder.fit_transform(data_test['education']) 
data_test['contact']=label_encoder.fit_transform(data_test['contact']) 
data_test['poutcome']=label_encoder.fit_transform(data_test['poutcome']) 
data_test['default']=data_test['default'].map({'yes':1,'no':0})
data_test['housing']=data_test['housing'].map({'yes':1,'no':0})
data_test['loan']=data_test['loan'].map({'yes':1,'no':0})
   


# In[133]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix,classification_report,recall_score,f1_score,accuracy_score,precision_score


# In[134]:


X_train=data.drop('Bank_Deposit',axis=1)
y_train=data['Bank_Deposit']


# In[135]:


X_test=data_test.drop('Bank_Deposit',axis=1)
y_test=data_test['Bank_Deposit']


#                      Logistic Regression

# In[136]:


mysteps = [('ss',StandardScaler()),('rus',RandomUnderSampler(random_state = 1)),('lr',LogisticRegression(solver='saga'))]

m_pipe = Pipeline(steps = mysteps)

m_pipe.fit(X_train,y_train)


# In[137]:


tr_Xtest=m_pipe['ss'].transform(X_test)


# In[138]:


predict_lr=m_pipe['lr'].predict(tr_Xtest)


# In[139]:


recall_lr=recall_score(y_test,predict_lr,pos_label='yes')
recall_lr


# In[140]:


f1_lr=f1_score(y_test, predict_lr, average='weighted')
f1_lr


# In[141]:


cm=confusion_matrix(y_test,predict_lr)
cm_matrix = pd.DataFrame(data=cm, columns=['no','yes'],
                        index=['no','yes'])
sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Confusion Matrix')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.show()


# In[61]:


from collections import Counter
print(Counter(y_test))


#                       Support Vector Machine     

# In[53]:


mysteps = [('ss',StandardScaler()),('rus',RandomUnderSampler(random_state = 1)),('svm', SVC(kernel = 'rbf'))]

m_pipe = Pipeline(steps = mysteps)

m_pipe.fit(X_train, y_train)


# In[54]:


tr_Xtest=m_pipe['ss'].transform(X_test)


# In[56]:


predict_svm=m_pipe['svm'].predict(tr_Xtest)


# In[57]:


recall_svm=recall_score(y_test,predict_svm,pos_label='yes')
recall_svm


# In[58]:


f1_svm=f1_score(y_test, predict_svm, average='weighted')
f1_svm


# In[60]:


cm=confusion_matrix(y_test,predict_svm)
cm_matrix = pd.DataFrame(data=cm, columns=['no','yes'],
                        index=['no','yes'])
sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Confusion Matrix')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.show()


#                            Decision Tree

# In[142]:


mysteps = [('rus',RandomUnderSampler(random_state = 1)),('dt',DecisionTreeClassifier())]

m_pipe = Pipeline(steps = mysteps)

m_pipe.fit(X_train, y_train)


# In[143]:


pre_dt=m_pipe['dt'].predict(X_test)


# In[144]:


recall_score(y_test,pre_dt,pos_label='yes')


# In[148]:


f1_dt=f1_score(y_test, pre_dt, average='weighted')
f1_dt


# In[149]:


cm=confusion_matrix(y_test,pre_dt)
cm_matrix = pd.DataFrame(data=cm, index=['no','yes'],
                        columns=['no','yes'])
sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Confusion Matrix')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.show()


#                         Extra Trees Classifier

# In[71]:


mysteps = [('rus',RandomUnderSampler(random_state = 1)),('etc',ExtraTreesClassifier())]

m_pipe = Pipeline(steps = mysteps)

m_pipe.fit(X_train, y_train)


# In[72]:


pre_etc=m_pipe['etc'].predict(X_test)


# In[77]:


recall_score(y_test,pre_etc,pos_label='yes')


# In[150]:


f1_etc=f1_score(y_test, pre_etc, average='weighted')
f1_etc


# In[75]:


cm=confusion_matrix(y_test,pre_etc)
cm_matrix = pd.DataFrame(data=cm, index=['no','yes'],
                        columns=['no','yes'])
sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Confusion Matrix')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.show()


#                       Random Forest Classifier

# In[79]:


mysteps = [('rus',RandomUnderSampler(random_state = 1)),('rf',RandomForestClassifier())]

m_pipe = Pipeline(steps = mysteps)

m_pipe.fit(X_train, y_train)


# In[80]:


pre_rf=m_pipe['rf'].predict(X_test)


# In[82]:


re_score=recall_score(y_test,pre_rf,pos_label='yes')
re_score


# In[151]:


f1_rf=f1_score(y_test, pre_rf, average='weighted')
f1_rf


# In[83]:


cm=confusion_matrix(y_test,pre_rf)
cm_matrix = pd.DataFrame(data=cm, index=['no','yes'],
                        columns=['no','yes'])
sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Confusion Matrix')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.show()


#                       Gaussian NB

# In[85]:


mysteps = [('rus',RandomUnderSampler(random_state = 1)),('gnb',GaussianNB())]

m_pipe = Pipeline(steps = mysteps)

m_pipe.fit(X_train, y_train)


# In[86]:


pre_gnb=m_pipe['gnb'].predict(X_test)


# In[92]:


recall_gnb_score=recall_score(y_test,pre_gnb,pos_label='yes')
recall_gnb_score


# In[152]:


f1_gnb=f1_score(y_test, pre_gnb, average='weighted')
f1_gnb


# In[90]:


cm=confusion_matrix(y_test,pre_gnb)
cm_matrix = pd.DataFrame(data=cm, index=['no','yes'],
                        columns=['no','yes'])
sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Confusion Matrix')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.show()


#                      Gradient Boosting Classifier               

# In[93]:


mysteps = [('rus',RandomUnderSampler(random_state = 1)),('gbc',GradientBoostingClassifier())]

m_pipe = Pipeline(steps = mysteps)

m_pipe.fit(X_train, y_train)


# In[94]:


pre_gbc=m_pipe['gbc'].predict(X_test)


# In[96]:


recall_gbc_score=recall_score(y_test,pre_gbc,pos_label='yes')
recall_gbc_score


# In[153]:


f1_gbc=f1_score(y_test, pre_gbc, average='weighted')
f1_gbc


# In[97]:


cm=confusion_matrix(y_test,pre_gbc)
cm_matrix = pd.DataFrame(data=cm, index=['no','yes'],
                        columns=['no','yes'])
sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Confusion Matrix')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.show()


#                           AdaBoost Classifier

# In[98]:


mysteps = [('rus',RandomUnderSampler(random_state = 1)),('abc',AdaBoostClassifier(base_estimator = LogisticRegression(max_iter = 10000), n_estimators = 100, learning_rate = 0.3))]

m_pipe = Pipeline(steps = mysteps)

m_pipe.fit(X_train, y_train)


# In[101]:


pre_abc=m_pipe['abc'].predict(X_test)


# In[102]:


recall_abc_score=recall_score(y_test,pre_abc,pos_label='yes')
recall_abc_score


# In[154]:


f1_abc=f1_score(y_test, pre_abc, average='weighted')
f1_abc


# In[103]:


cm=confusion_matrix(y_test,pre_abc)
cm_matrix = pd.DataFrame(data=cm, columns=['No','Yes'],
                        index=['No','Yes'])
sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Confusion Matrix')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.show()


#                         XGB Boosting

# In[155]:


y_train=data['Bank_Deposit'].map({'yes':1,'no':0})


# In[156]:


y_test=data_test['Bank_Deposit'].map({'yes':1,'no':0})


# In[157]:


mysteps = [('rus',RandomUnderSampler(random_state = 1)),('XGB',XGBClassifier())]

m_pipe = Pipeline(steps = mysteps)

m_pipe.fit(X_train, y_train)


# In[158]:


y_xgb_predict=m_pipe['XGB'].predict(X_test)


# In[159]:


re_score=recall_score(y_test,y_xgb_predict,pos_label=1)
re_score


# In[160]:


f1_xgb=f1_score(y_test, y_xgb_predict, average='weighted')
f1_xgb


# In[110]:


cm=confusion_matrix(y_test,y_xgb_predict)
cm_matrix = pd.DataFrame(data=cm, columns=['0','1'],
                        index=['0','1'])
sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Confusion Matrix')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.show()


# In[ ]:





# In[164]:


comparison={'Models':['Logistic Regreesion', 'Support Vector Machine','Decision Tree','ExtraTrees','Random Forest','Gaussian NB','Gradient Boost','Ada Boost','XGB'],
            'Recall_Score':[0.781190019193858,0.8445297504798465,1.0,1.0,1.0,0.8157389635316699,0.8426103646833013,0.744721689059501,0.9251439539347409],
           'f1_Score':[0.8231198933342957,0.8276885369271005,0.8347850167640332,0.8604141354177849,0.8636841845108688,0.7457476107037891,0.835086412178487,0.8268485819336772,0.8528730676013049]}


# In[165]:


df=pd.DataFrame(comparison)
df.sort_values(by=['f1_Score'],ascending=False,ignore_index=True)


# Among the all the models, Random Forest works better for this Banking Dataset.. This dataset is an imbalanced dataset with binary classes as yes and no... All the tree based models have recall score 1 for the minority class i.e yes class. All the positives are predicted perfectly by the tree based models. 
