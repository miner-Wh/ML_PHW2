from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN

from sklearn.cluster import estimate_bandwidth
from sklearn.cluster import MeanShift
import math
import time
import warnings ; warnings.filterwarnings('ignore')
import seaborn as sns
from sklearn.metrics import accuracy_score, silhouette_samples, silhouette_score
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, StratifiedKFold 
skf = StratifiedKFold(n_splits=10) 
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import sys

from sklearn.cluster import KMeans

################
#make result table
score_sample = {'Group':["Group"],'Scaler':["Sample"], 'Encoder':["Sample"], 'Model':["Sample"],'Best_para':["Sample"], "Score":[1]}
score_results = pd.DataFrame(score_sample)

score_sample2 = {'type':["eeror"],'info':["info"]}
error_data = pd.DataFrame(score_sample2)




#for scale and encorde
class PreprocessPipeline(): 
    def __init__(self, num_process, cat_process, verbose=False): 
        #super(PreprocessPipeline, self).__init__() 
        self.num_process = num_process 
        self.cat_process = cat_process 
        #for each type
        if num_process == 'standard': 
            self.scaler = preprocessing.StandardScaler() 
        elif num_process == 'minmax': 
            self.scaler = preprocessing.MinMaxScaler() 
        elif num_process == 'maxabs': 
            self.scaler = preprocessing.MaxAbsScaler() 
        elif num_process == 'robust': 
            self.scaler = preprocessing.RobustScaler() 
        else: 
            raise ValueError("Supported 'num_process' : 'standard','minmax','maxabs','robust'")   
        if cat_process == 'onehot': 
            self.encoder = preprocessing.OneHotEncoder(sparse=False, handle_unknown='ignore')  
        elif cat_process == 'ordinal': 
            self.encoder = preprocessing.OrdinalEncoder() 
        else: 
            raise ValueError("Supported 'cat_process' : 'onehot', ordinal'") 

        self.verbose=verbose 
        
        #do Preprocess
    def process(self, X): 
        X_cats = X.select_dtypes(np.object).copy() 
        X_nums = X.select_dtypes(exclude=np.object).copy() 
        #Xt_cats = Xt.select_dtypes(np.object).copy() 
        #Xt_nums = Xt.select_dtypes(exclude=np.object).copy() 

        if self.verbose: 
            print(f"Categorica Colums : {list(X_cats)}") 
            print(f"Numeric Columns : {list(X_nums)}") 

        if self.verbose: 
            print(f"Categorical cols process method : {self.cat_process.upper()}") 

        X_cats = self.encoder.fit_transform(X_cats) 
        #Xt_cats = self.encoder.transform(Xt_cats) 

        if self.verbose: 
            print(f"Numeric columns process method : {self.num_process.upper()}") 
        X_nums = self.scaler.fit_transform(X_nums) 
        #Xt_nums = self.scaler.transform(Xt_nums) 

        X_processed = np.concatenate([X_nums, X_cats],1) 
        #Xt_processed = np.concatenate([Xt_nums, Xt_cats], axis=-1) 

        return X_processed

# do process on I want 
class AutoProcess():
    def __init__(self, verbose=False):
        
        self.pp = PreprocessPipeline
        self.verbose= verbose
    
    def run(self, X,group):
        methods = []
        scores = []
        print(X.shape)
        
        for num_process in ['standard','robust','minmax','maxabs']:
            for cat_process in ['ordinal','onehot']:
                if self.verbose:
                    print("\n------------------------------------------------------\n")
                    print(f"Numeric Process : {num_process}")
                    print(f"Categorical Process : {cat_process}")
                methods.append([num_process, cat_process])

                pipeline = self.pp(num_process=num_process, cat_process=cat_process)
                
                X_processed= pipeline.process(X)
                
                #print(X_processed.shape)
                #Classifier part
                for model in ['k-mean','em','clarans','dbscan','mean-shift']:
                    if self.verbose:
                        print(f"\nCluster model: {model}")

                    if model =='k-mean': 
                        k_num = {2,3,4,5,7,10}
                        for k in k_num:
                            c_mdel = KMeans(n_clusters=k)
                            #print(X_processed)
                            c_mdel.fit(X_processed)
                            sample = X.copy()
                            sample['cluster'] = c_mdel.labels_
                            sample_score = silhouette_samples(X_processed,sample['cluster'] )
                            sample['silhouette_'] = sample_score
                            score_results.loc[len(score_results)] = [group, num_process, cat_process, model,'k='+str(k), str(sample['silhouette_'].mean())]

                    elif model == 'em':
                        k_num ={2,3,4,5,7,10}
                        for k in k_num:
                            c_mdel = GaussianMixture(n_components=k, random_state=0).fit(X_processed)
                            sample = X.copy()
                            c_mdel_cluster_labels = c_mdel.predict(X_processed)
                            sample['cluster'] = c_mdel_cluster_labels
                            sample_score = silhouette_samples(X_processed,sample['cluster'])
                            sample['silhouette_'] = sample_score
                            score_results.loc[len(score_results)] = [group, num_process, cat_process, model,'k='+str(k), str(sample['silhouette_'].mean())]

                   

                    elif model == 'dbscan':
                        esp = {0.01, 0.1, 0.2, 0.3, 0.5, 0.75 }
                        ms = {2,3,5,7,10}
                        for e in esp:
                            for m in ms:
                                try:
                                    c_mdel = DBSCAN(eps = e, min_samples=m)
                                    sample = X.copy()
                                    sample['cluster'] = pd.DataFrame(c_mdel.fit_predict(X_processed))
                                    sample_score = silhouette_samples(X_processed,sample['cluster'])
                                    sample['silhouette_'] = sample_score
                                    score_results.loc[len(score_results)] = [group, num_process, cat_process, model,'eps: '+str(e)+'  m: '+str(m)+'  cluster: '+str(len(sample['cluster'].value_counts())), str(sample['silhouette_'].mean())]
                                except ValueError:
                                    error_data.loc[len(error_data)]=['ValueError','eps: '+str(e)+'  ms: '+str(m)+'only one cluster']

                    elif model == 'mean-shift':
                        best_bandwidth = estimate_bandwidth(X_processed)
                        c_mdel = MeanShift(bandwidth=best_bandwidth)
                        c_mdel_cluster_labels = c_mdel.fit_predict(X_processed)
                        sample = X.copy()
                        sample['cluster'] = c_mdel_cluster_labels

                        print('cluster labels type: ', np.unique(c_mdel_cluster_labels))
                        print('bandwidth값 : ',best_bandwidth)
                        sample_score = silhouette_samples(X_processed,sample['cluster'])
                        sample['silhouette_'] = sample_score
                        print('aver sihouette_: ' +str(sample['silhouette_'].mean()))
                        print(sample.groupby('cluster')['silhouette_'].mean())
                        score_results.loc[len(score_results)] = [group, num_process, cat_process, model,'bandwidth: '+str(best_bandwidth), str(sample['silhouette_'].mean())]
                    

        return

kfold = KFold(5, True, 1)

pd.set_option('display.max_row', 10000)
# Import the data file
df = pd.read_csv('C:\Users\Minner\Documents\VSC_workS\ML_PHW2\housing.csv', encoding='utf-8')
print(df.dtypes)
print(df.isna().sum())

##setting data set
#ex = df.iloc[:,8]
df = df.drop('median_house_value',axis=1)
#df['median_house_value'] = ex.to_numpy()

#fill na in total_bedrooms
#I thought it was marked Na because the value is 0.
df = df.fillna(0)
print(df.isna().sum())
#group 1 room
X1 = df[['total_rooms','total_bedrooms']]

#group 2 where
X2 = df[['longitude','latitude','ocean_proximity']]

#group 3 spec
X3 = df[['housing_median_age','total_rooms','total_bedrooms','ocean_proximity']]

#group 4 eviroment
X4 = df[['population','households']]

#group 5 all
X5 = df
autoprocess = AutoProcess(verbose=True)
autoprocess.run(X1,'room')
autoprocess.run(X2,'where')
autoprocess.run(X3,'spec')
autoprocess.run(X4,'eviroment')
autoprocess.run(X5,'all')

print(score_results)

sys.stdout = open('E:\PythonWorkSpace\score result.txt', 'w')

print(score_results)

sys.stdout.close()


