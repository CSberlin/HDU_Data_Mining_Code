#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as mp
import math
import os
from sklearn import preprocessing
from scipy.io import arff
import warnings
warnings.filterwarnings('ignore')


# In[4]:


def Data_Load(data_path):
    data =  arff.loadarff('.//Data//' + data_path)
    df = pd.DataFrame(data[0])
    data = np.array(df)
    m,n=data.shape
    return data,m,n


# In[5]:


def compute_N_Y_k(data):
    list_NY = []
    list_data = list(data[:,-1])
    N = list_data.count(b'N')
    Y = list_data.count(b'Y')
    k = math.floor(N/Y)
    return N,Y,k


# In[31]:


def compute_scale(data,m,N,Y):
   data_scaled=preprocessing.scale(data[:,:-1], axis=0, with_mean=True,with_std=True,copy=True) 
   data_scaled_NY = np.concatenate((data_scaled,data[:,-1].reshape(m,1)),axis=1)
   index = np.arange(m)
   data_scaled = np.concatenate((index.reshape(m,1),data_scaled_NY),axis=1)
   data_scaled_N_index  = np.argwhere(data_scaled_NY[:,-1]==b'N')
   data_scaled_Y_index  = np.argwhere(data_scaled_NY[:,-1]==b'Y')
   data_scaled_N = data_scaled[data_scaled_N_index].reshape(N,n+1)
   data_scaled_Y = data_scaled[data_scaled_Y_index].reshape(Y,n+1)
   
   return data_scaled_N,data_scaled_Y


# In[7]:


def Cal_Distance(data_scaled_N,data_scaled_Y,N,Y,k):
    NY_distance_samples_neighbor = np.empty([0,3],float)
    flag = 0
    i=0
    for s_Y in data_scaled_Y:
        temp_arr = np.array([])
        for s_N in data_scaled_N:
            temp = np.linalg.norm(s_N[1:n]-s_Y[1:n])
            temp_arr = np.append(temp_arr,temp)

        temp_arr = temp_arr.reshape((N,1))
        NY_distance = np.concatenate((data_scaled_Y[[i],[0]].repeat(N).reshape(N,1),data_scaled_N[:,[0]],temp_arr),axis=1)
        NY_distance_sample_neighbor = NY_distance[NY_distance[:,2].argsort()].reshape(N,3)[0:k,:]
        NY_distance_samples_neighbor = np.append(NY_distance_samples_neighbor,NY_distance_sample_neighbor,axis=0)
        i = i+1
    return NY_distance_samples_neighbor


# In[8]:


def Cal_Feature_Differential(data,NY_distance_samples_neighbor,m,n):
    w_matrix = np.zeros(n-1).reshape(1,n-1)
    NY_M,NY_N = NY_distance_samples_neighbor.shape
    index_ny = NY_distance_samples_neighbor[0:NY_M,0:NY_N-1]

    index_y = index_ny[0:NY_M,0].reshape(NY_M).astype(int)
    index_n = index_ny[0:NY_M,1].reshape(NY_M).astype(int)

    index_last = np.arange(m)
    data_filter = np.concatenate((index_last.reshape(m,1),data.copy()),axis=1)

    samples_feature_differential = np.abs(data_filter[list(index_y),1:n]-data_filter[list(index_n),1:n])

    feature_index = np.arange(n-1).reshape(1,n-1)
    
    for feature_differential in samples_feature_differential:
        temp_feature = np.concatenate((feature_index,feature_differential.reshape(1,n-1)),axis=0)
        temp_sorted_feature = temp_feature[:,temp_feature[1].argsort()]
        feature = temp_sorted_feature[1,:].copy()
        j = 1 
        temp_w = np.empty(n-1)
        temp_w[0] = j
        for i in range(n-1):
            if i!=0 :
                if  feature[i-1]==feature[i]:
                    temp_w[i] = j
                else:
                    j = j+1
                    temp_w[i] = j
        temp_sorted_feature[1,:]=temp_w
        temp_last_feature  = temp_sorted_feature[:,temp_sorted_feature[0].argsort()]
        w_matrix = np.add(temp_last_feature[1,:],w_matrix)
        
    index = np.arange(n-1).reshape(1,n-1)
    last = np.concatenate((index,w_matrix),axis=0)
    last = last[:,last[1].argsort()][::,::-1]
    return last


# In[9]:


from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
def Train_Measure(X,y,clf):
    auc1 = np.empty(100)
    count = 0
    rkf = RepeatedKFold(n_splits=10,n_repeats=10)
    for train_index,test_index in rkf.split(X):
        x_train,x_test = X[train_index],X[test_index]
        y_train,y_test = y[train_index],y[test_index]
        clf.fit(x_train,y_train)
        pre_y = clf.predict_proba(x_test)[:,1]
        fpr,tpr,thresholds = roc_curve(y_test,pre_y)
        auc1[count] = auc(fpr,tpr)
        count += 1
    auc1 = np.nanmean(auc1)
    return auc1


# In[10]:


from sklearn import svm
from sklearn import naive_bayes
from sklearn.tree import DecisionTreeClassifier
def Model_Initialize():
    clf_svm = svm.SVC(gamma='auto',probability=True)
    clf_bayes = naive_bayes.GaussianNB()
    clf_tree = DecisionTreeClassifier()
    return clf_svm,clf_bayes,clf_tree


# In[32]:



import os
if __name__ == '__main__':
    files = os.listdir('D:\Code\Data_Mining_Code\ex3\Data')
    for file in files:
        
        Original_Data,m,n = Data_Load(file)
        N,Y,k = compute_N_Y_k(Original_Data)
        data_scaled_N,data_scaled_Y = compute_scale(Original_Data,m,N,Y)
        NY_distance_samples_neighbor = Cal_Distance(data_scaled_N,data_scaled_Y,N,Y,k)
        Feature_Sequence = Cal_Feature_Differential(Original_Data,NY_distance_samples_neighbor,m,n).astype(np.int32)
        data_scaled_N_Y = np.concatenate((data_scaled_N,data_scaled_Y),axis=0)
        data_scaled_N_Y = data_scaled_N_Y[data_scaled_N_Y[:,0].argsort()]
        data = data_scaled_N_Y[:,Feature_Sequence[0,:]+1]
        #将N,Y变换为0，1
        lookupTable,label = np.unique(data_scaled_N_Y[:,n], return_inverse=True)
        data = np.concatenate((data,label.reshape(-1,1)),axis=1)
        if os.path.exists(file+'.txt'):
            continue
        else:
            np.savetxt(file+'.txt',data)

        if os.path.exists(file+'.csv'):
            continue
        data = np.loadtxt('D:\Code\Data_Mining_Code\ex3\\txtdata\\'+str(file)+'.txt')
        m,n = data.shape
        print(file)
        d = int(math.log(N+Y,2))
        print("d: "+ str(d))

        clfs = Model_Initialize()
        Feature_index = list(range(1,n+1))
        roc_f1_columns = ['roc_auc_svm','roc_auc_byes','roc_auc_tree'] 
        auc_Array = np.array([])
        for i in range(n):
            print("Feature 1 To "+str(i+1))
            for clf in clfs:    
                mean_auc = Train_Measure(data[:,0:i+1],data[:,-1],clf)
                auc_Array= np.append(auc_Array,mean_auc)
            auc_Array = auc_Array.reshape(-1,3)
        df = pd.DataFrame(auc_Array,index=Feature_index,columns=roc_f1_columns)
        print(file+'   '+'_roc_auc:  ')
        print(df)
        df.to_csv(file+'.csv')

# In[ ]:




