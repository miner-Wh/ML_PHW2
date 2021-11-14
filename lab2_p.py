from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn import preprocessing
import warnings ; warnings.filterwarnings('ignore')
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import sys
from pyclustering.cluster.clarans import clarans
from sklearn.cluster import KMeans, estimate_bandwidth, MeanShift
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN

################
#make result table
score_sample = {'Group':["Group"],'Scaler':["Sample"], 'Encoder':["Sample"], 'Model':["Sample"],'Best_para':["Sample"], "Score":[1]}
score_results = pd.DataFrame(score_sample)
score_sample2 = {'type':["eeror"],'info':["info"]}
error_data = pd.DataFrame(score_sample2)



#for scale and encorde
class PreprocessPipeline():
    def __init__(self, num_process, cat_process, verbose=False):
        # super(PreprocessPipeline, self).__init__()
        self.num_process = num_process
        self.cat_process = cat_process
        # for each type
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

        self.verbose = verbose

        # do Preprocess

    def process(self, X):
        X_cats = X.select_dtypes(np.object).copy()
        X_nums = X.select_dtypes(exclude=np.object).copy()
        # Xt_cats = Xt.select_dtypes(np.object).copy()
        # Xt_nums = Xt.select_dtypes(exclude=np.object).copy()

        if self.verbose:
            print(f"Categorica Colums : {list(X_cats)}")
            print(f"Numeric Columns : {list(X_nums)}")

        if self.verbose:
            print(f"Categorical cols process method : {self.cat_process.upper()}")

        X_cats = self.encoder.fit_transform(X_cats)
        # Xt_cats = self.encoder.transform(Xt_cats)

        if self.verbose:
            print(f"Numeric columns process method : {self.num_process.upper()}")
        X_nums = self.scaler.fit_transform(X_nums)
        # Xt_nums = self.scaler.transform(Xt_nums)

        X_processed = np.concatenate([X_nums, X_cats], 1)
        # Xt_processed = np.concatenate([Xt_nums, Xt_cats], axis=-1)

        return X_processed

# do process on I want 


class AutoProcess():
    def __init__(self, verbose=False):

        self.pp = PreprocessPipeline
        self.verbose = verbose

    def run(self, X, group):
        methods = []
        scores = []
        # print(X.shape)

        # need dataframe, list of label, number of clusters
        def makePlt23(df, label, k, count, title):
            if(len(df.columns)==3):
                enc=preprocessing.OrdinalEncoder()
                op = enc.fit_transform(df['ocean_proximity'].to_numpy().reshape(-1, 1))
                df['ocean_proximity'] = op
            # list for store feature data for each cluster
            store = [[[] for col in range(len(df.columns))] for row in range(k)]

            for m in range(len(label)):
                for n in range(k):
                    if (label[m] == n):
                        for o in range(len(df.columns)):
                            store[n][o].append(df.iloc[m:m + 1, o:o + 1].values[0][0])

            c = ['b.', 'r.', 'g.', 'y.', 'c.', 'm.', 'k.']
            if (len(df.columns) == 2):
                plt.subplot(120 + count, title=title)
                plt.xlabel(df.columns[0])
                plt.ylabel(df.columns[1])
                for p in range(k):
                    plt.plot(store[p][0], store[p][1], c[p])
            if (len(df.columns) == 3):
                plt.subplot(120 + count, projection='3d', title=title)
                plt.xlabel(df.columns[0])
                plt.ylabel(df.columns[1])
                for p in range(k):
                    plt.plot(store[p][0], store[p][1], store[p][2], c[p])

        def makePltBig(df, label, k):
            if (len(df.columns) == 3):
                enc = preprocessing.OrdinalEncoder()
                op = enc.fit_transform(df['ocean_proximity'].to_numpy().reshape(-1, 1))
                df['ocean_proximity'] = op
            # list for store feature data for each cluster
            store = [[[] for col in range(len(df.columns))] for row in range(k)]

            for m in range(len(label)):
                for n in range(k):
                    if (label[m] == n):
                        for o in range(len(df.columns)):
                            store[n][o].append(df.iloc[m:m + 1, o:o + 1].values[0][0])

            if(len(df.columns)==2):
                for j in range(int(k/9)+1):
                    for i in range(j*9, (j+1)*9, 1):
                        if(i<k):
                            plt.subplot(330+(i-(j*9)+1), title='Cluster N.'+str(i+1))
                            plt.xlabel(df.columns[0])
                            plt.ylabel(df.columns[1])
                            plt.plot(store[i][0], store[i][1], '.')
                    plt.show()
            if (len(df.columns) == 3):
                for j in range(int(k / 9) + 1):
                    for i in range(j * 9, (j + 1) * 9, 1):
                        if (i < k):
                            plt.subplot(330 + (i - (j * 9) + 1), projection='3d', title='Cluster N.' + str(i + 1))
                            plt.xlabel(df.columns[0])
                            plt.ylabel(df.columns[1])
                            plt.plot(store[i][0], store[i][1], store[i][2], '.')
                    plt.show()

        for num_process in ['maxabs']:
            for cat_process in ['ordinal']:
                if self.verbose:
                    print("\n------------------------------------------------------\n")
                    print(f"Numeric Process : {num_process}")
                    print(f"Categorical Process : {cat_process}")
                methods.append([num_process, cat_process])

                pipeline = self.pp(num_process=num_process, cat_process=cat_process)

                X_processed = pipeline.process(X)

                # print(X_processed.shape)
                # Classifier part
                for model in ['k-mean', 'em', 'clarans', 'dbscan', 'mean-shift']:
                    if self.verbose:
                        print(f"\nCluster model: {model}")

                    if model == 'k-mean':
                        if group == 'room':
                            k_num = {3, 5}
                        elif group == 'where':
                            k_num = {4, 7}
                        elif group == 'eviroment':
                            k_num = {5, 7}

                        countPlt=0
                        for k in k_num:
                            countPlt=countPlt+1
                            c_mdel = KMeans(n_clusters=k)
                            # print(X_processed)
                            c_mdel.fit(X_processed)
                            sample = X.copy()
                            sample['cluster'] = c_mdel.labels_
                            sample_score = silhouette_samples(X_processed, sample['cluster'])
                            sample['silhouette_'] = sample_score
                            score_results.loc[len(score_results)] = [group, num_process, cat_process, model,
                                                                     'k=' + str(k), str(sample['silhouette_'].mean())]
                            print(group, num_process, cat_process, model, 'k=' + str(k),
                                  str(sample['silhouette_'].mean()))
                            print(sample.groupby('cluster')['silhouette_'].mean())
                            makePlt23(X, sample['cluster'], k, countPlt, 'k='+str(k))
                        plt.show()

                    if model == 'em':
                        if group == 'room':
                            k_num = {3, 5}
                        elif group == 'where':
                            k_num = {5, 7}
                        elif group == 'eviroment':
                            k_num = {3, 5}

                        countPlt = 0
                        for k in k_num:
                            countPlt+=1
                            c_mdel = GaussianMixture(n_components=k, random_state=0).fit(X_processed)
                            sample = X.copy()
                            c_mdel_cluster_labels = c_mdel.predict(X_processed)
                            sample['cluster'] = c_mdel_cluster_labels
                            sample_score = silhouette_samples(X_processed, sample['cluster'])
                            sample['silhouette_'] = sample_score
                            score_results.loc[len(score_results)] = [group, num_process, cat_process, model,
                                                                     'k=' + str(k), str(sample['silhouette_'].mean())]
                            print(group, num_process, cat_process, model, 'k=' + str(k),
                                  str(sample['silhouette_'].mean()))
                            print(sample.groupby('cluster')['silhouette_'].mean())
                            makePlt23(X, sample['cluster'], k, countPlt, 'k='+str(k))
                        plt.show()

                    if model == 'clarans':
                        if group == 'room':
                            k_num = {3, 5}
                        elif group == 'where':
                            k_num = {5, 7}
                        elif group == 'eviroment':
                            k_num = {3, 5}

                        countPlt=0
                        for k in k_num:
                            countPlt+=1
                            sample = X[1400:2400].copy()

                            # make list to store each rows label
                            label = [0 for l in range(len(X[1400:2400]))]

                            # data, number of cluster, num local, max neighbor
                            clarans_instance = clarans(X_processed[1400:2400], k, 6, 4)
                            clarans_instance.process()
                            clusters = clarans_instance.get_clusters()

                            # make label
                            for j in range(0, len(clusters), 1):
                                for i in range(0, len(clusters[j]), 1):
                                    label[clusters[j][i]] = j
                            sample['cluster']=label

                            sample_score = silhouette_samples(X_processed[1400:2400], sample['cluster'])
                            # print(k, 'clusters silhouette score :', score)
                            sample['silhouette_']=sample_score
                            score_results.loc[len(score_results)] = [group, num_process, cat_process, model, 'k=' + str(k), str(sample['silhouette_'].mean())]
                            print(group, num_process, cat_process, model, 'k=' + str(k), str(sample['silhouette_'].mean()))
                            makePlt23(X[1400:2400], label, k, countPlt, 'k='+str(k))
                        plt.show()

                    if model == 'dbscan':
                        if group == 'room':
                            esp = {0.01}
                            ms = {3, 5}
                        elif group == 'where':
                            esp = {0.01, 0.75}
                            ms = {7, 10}
                        elif group == 'eviroment':
                            esp = {0.01}
                            ms = {3, 5}
                        for e in esp:
                            countPlt=0
                            for m in ms:
                                countPlt+=1
                                try:
                                    c_mdel = DBSCAN(eps=e, min_samples=m)
                                    sample = X.copy()
                                    sample['cluster'] = pd.DataFrame(c_mdel.fit_predict(X_processed))
                                    sample_score = silhouette_samples(X_processed, sample['cluster'])
                                    sample['silhouette_'] = sample_score
                                    score_results.loc[len(score_results)] = [group, num_process, cat_process, model,
                                                                             'eps: ' + str(e) + '  m: ' + str(
                                                                                 m) + '  cluster: ' + str(
                                                                                 len(sample['cluster'].value_counts())),
                                                                             str(sample['silhouette_'].mean())]
                                    print(group, num_process, cat_process, model,
                                          'eps: ' + str(e) + '  m: ' + str(m) + '  cluster: ' + str(
                                              len(sample['cluster'].value_counts())), str(sample['silhouette_'].mean()))
                                    sample['cluster'] = sample['cluster'] + 1
                                    print(sample.groupby('cluster')['silhouette_'].mean())
                                    print(sample['cluster'].value_counts())
                                    # sample['cluster']=sample['cluster']+1
                                    k=len(sample[['cluster']].groupby('cluster').count())
                                    if(k<=7):
                                        makePlt23(X, sample['cluster'], k, countPlt, 'eps='+str(e)+' m='+str(m))
                                    else:
                                        makePltBig(X, sample['cluster'], k)


                                except ValueError:
                                    error_data.loc[len(error_data)] = ['ValueError', 'eps: ' + str(e) + '  ms: ' + str(
                                        m) + 'only one cluster']
                            plt.show()

                    elif model == 'mean-shift':
                        best_bandwidth = estimate_bandwidth(X_processed)
                        c_mdel = MeanShift(bandwidth=best_bandwidth)
                        c_mdel_cluster_labels = c_mdel.fit_predict(X_processed)
                        sample = X.copy()
                        sample['cluster'] = c_mdel_cluster_labels

                        print('cluster labels type: ', np.unique(c_mdel_cluster_labels))
                        print('bandwidth값 : ', best_bandwidth)
                        sample_score = silhouette_samples(X_processed, sample['cluster'])
                        sample['silhouette_'] = sample_score
                        print('aver sihouette_: ' + str(sample['silhouette_'].mean()))
                        print(sample.groupby('cluster')['silhouette_'].mean())
                        score_results.loc[len(score_results)] = [group, num_process, cat_process, model,
                                                                 'bandwidth: ' + str(best_bandwidth),
                                                                 str(sample['silhouette_'].mean())]
                        print(group, num_process, cat_process, model, 'bandwidth: ' + str(best_bandwidth),
                              str(sample['silhouette_'].mean()))
                        print(sample.groupby('cluster')['silhouette_'].mean())
                        k = len(sample[['cluster']].groupby('cluster').count())
                        if(k<=7):
                            makePlt23(X, sample['cluster'], k, 1, 'bandwidth='+str(best_bandwidth))
                        else:
                            makePltBig(X, sample['cluster'], k)
                        plt.show()

        return



# Import the data file
df = pd.read_csv('C:\Users\Minner\Documents\VSC_workS\ML_PHW2\housing.csv', encoding='utf-8')
# print(df.dtypes)
# print(df.isna().sum())

##setting data set
# separate median house value feature
mhv=df['median_house_value']
df.drop('median_house_value',axis=1, inplace=True)
# fill nan value in total_bedrooms
df.fillna(0, inplace=True)
# print(df.isna().sum())

#group 1 room
X1 = df[['total_rooms','total_bedrooms']]

#group 2 where
X2 = df[['longitude','latitude','ocean_proximity']]

#group 4 eviroment
X4 = df[['population','households']]

autoprocess = AutoProcess(verbose=True)
autoprocess.run(X1,'room')
autoprocess.run(X2,'where')
autoprocess.run(X4,'eviroment')
print(score_results)
