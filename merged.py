#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 12:49:02 2022

@author: alex
"""

randomSeed = None
decisionThreshold = 0.5
cutOutlierThreshold = 5
#%%

#Take Data
import pandas as pd

data = pd.read_excel('training.xlsx', engine='openpyxl')
officialTest = pd.read_excel('test.xlsx', engine='openpyxl').drop(columns=['Y-Prediction']).dropna()

#%%
#data exploration
    
def exploration():
    import matplotlib.pyplot as plt
    import numpy as np
    
    #count na
    print(data.isna().sum())
    
    classDist = data.groupby(by=['Y'], dropna=True).size()
    
    #pie of class distribution
    plt.pie(classDist, autopct=lambda p: '{:.0f}'.format(p * sum(classDist) / 100),
            labels = classDist.index)
    plt.title('class distribution')
    plt.show()
    
    
    import seaborn as sns
    sns.pairplot(data, height=5, hue='Y')
    plt.show()
    
    plt.figure(figsize=(8,6))
    corr = data.corr()
    sns.heatmap(corr, cmap="Blues",annot=False)
    plt.title('corr matrix')
    plt.show()


#%%
#Train Test Split
def split(columnsList):
    from sklearn.model_selection import train_test_split
    xT, xt, yT, yt = train_test_split(data[columnsList], data['Y'], test_size=0.3, random_state=randomSeed)
    return xT, xt, yT, yt

#%%
def removeOutlier(x,y):
    import numpy as np
    from scipy import stats
    
    x['Y'] = y
    x = x[(np.abs(stats.zscore(x)) < cutOutlierThreshold).all(axis=1)]
    y = x['Y']
    x = x.drop(columns=['Y'])
    return x,y
#%%
#addressing problems in data

#imbalance

def oversample(x,y):
    from imblearn.over_sampling import BorderlineSMOTE
    X_res, Y_res = BorderlineSMOTE(random_state=randomSeed, kind='borderline-2').fit_resample(x, y)
    return X_res, Y_res

def oversampleVanilla(x,y):
    from imblearn.over_sampling import RandomOverSampler
    X_res, Y_res = RandomOverSampler(random_state=randomSeed).fit_resample(x, y)
    return X_res, Y_res

def undersample(x,y):
    from imblearn.under_sampling import TomekLinks
    X_res, Y_res = TomekLinks().fit_resample(x, y)
    return X_res, Y_res

#%%
#Models
def randomForest(xT,yT,xt):
    from sklearn.metrics import accuracy_score
    from sklearn.ensemble import RandomForestClassifier
    
    clf = RandomForestClassifier(random_state=randomSeed, n_estimators=10, bootstrap=False)
    clf.fit(xT,yT)
    predT = clf.predict(xT)
    realT = yT
    trainAccuracy = accuracy_score(predT, realT)
    print("Accuracy: %.2f%%" % (trainAccuracy * 100.0))
    predicted = clf.predict(xt)
    proba = clf.predict_proba(xt)
    #print(predicted, proba)
    return predicted, proba, clf

def xgb(xT,yT,xt):
    from sklearn.metrics import accuracy_score
    from xgboost import XGBClassifier
    
    xgb = XGBClassifier()
    xgb.fit(xT,yT)
    predT = xgb.predict(xT)
    realT = yT
    trainAccuracy = accuracy_score(predT, realT)
    print("Accuracy: %.2f%%" % (trainAccuracy * 100.0))
    predicted = xgb.predict(xt)
    proba = xgb.predict_proba(xt)
    #print(predicted, proba)
    return predicted, proba, xgb

def logReg(xT, yT, xt):
    import statsmodels.api as sm
    import numpy as np
    from sklearn.metrics import accuracy_score

    xTX12 = pd.get_dummies(xT['X12'], prefix="X12_", prefix_sep=" ",drop_first = True)
    xT = pd.concat([xT,xTX12], axis =1 )
    xT = xT.drop(columns = ['X12','X11','X3'])
    sm.add_constant(xT)

    xtX12 = pd.get_dummies(xt['X12'], prefix="X12_", prefix_sep=" ",drop_first = True)
    xt = pd.concat([xt,xtX12], axis =1 )
    xt = xt.drop(columns = ['X12','X11','X3'])
    sm.add_constant(xt)
    
    logit_mod = sm.Logit(yT, xT)
    cutoff = 0.5 # multiple thresholds (0.3, 0.4, 0.6, 0.7) attempted to conclude to this best threshold.
    logit_mod = sm.Logit(yT,xT)
    logit_res = logit_mod.fit()
    
    predTproba = logit_res.predict(xT)
    predT = np.where(predTproba > cutoff, 1,0)
    realT = yT
    trainAccuracy = accuracy_score(predT, realT)
    print("Accuracy: %.2f%%" % (trainAccuracy * 100.0))

    predictedProba = logit_res.predict(xt)
    predicted = np.where(predictedProba > cutoff, 1,0)

    return predicted, np.c_[np.ones(predictedProba.shape[0]),predictedProba], logit_res

def qda(xT, yT, xt):
    import statsmodels.api as sm
    import numpy as np
    from sklearn.metrics import accuracy_score
    import sklearn.discriminant_analysis as da
    
    xTX12 = pd.get_dummies(xT['X12'], prefix="X12_", prefix_sep=" ",drop_first = True)
    xT = pd.concat([xT,xTX12], axis =1 )
    xT = xT.drop(columns = ['X12','X11','X3'])
    sm.add_constant(xT)

    xtX12 = pd.get_dummies(xt['X12'], prefix="X12_", prefix_sep=" ",drop_first = True)
    xt = pd.concat([xt,xtX12], axis =1 )
    xt = xt.drop(columns = ['X12','X11','X3'])
    sm.add_constant(xt)
    
    QDA = da.QuadraticDiscriminantAnalysis()
    model_QDA = QDA.fit(xT,yT)
    
    predT = model_QDA.predict(xT)
    realT = yT
    trainAccuracy = accuracy_score(predT, realT)
    print("Accuracy: %.2f%%" % (trainAccuracy * 100.0))
    
    predicted = model_QDA.predict(xt)
    predictedProba = model_QDA.predict_proba(xt)
    
    return predicted, predictedProba,model_QDA

def PCA(xT, xt):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    
    stdScaler = StandardScaler().fit(xT) #return mean and sd
    xT = stdScaler.transform(xT) #return sdscaler number
    xt = stdScaler.transform(xt)
    
    pca=PCA(n_components=2,svd_solver='auto').fit(xT)
    xT = pca.transform(xT)
    xt = pca.transform(xt)
    
    stdScaler = StandardScaler().fit(xT) #return mean and sd
    xT = stdScaler.transform(xT) #return sdscaler number
    xt = stdScaler.transform(xt)

    return xT, xt
    
def svm(xT, yT, xt):
    from sklearn.metrics import accuracy_score
    xT, xt = PCA(xT, xt)
    
    from sklearn.svm import SVC
    
    # before tuning
    # svc = SVC(kernel='linear', probability=True)
    # svc.fit(xT, yT)
    
    #gridSearchSVC(xT, yT)

    svc=SVC(C=1,kernel='rbf',gamma=1, probability=True)
    
    svc.fit(xT, yT)
    predT = svc.predict(xT)
    realT = yT
    trainAccuracy = accuracy_score(predT, realT)
    print("Accuracy: %.2f%%" % (trainAccuracy * 100.0))
    
    predicted = svc.predict(xt)
    predictedProba = svc.predict_proba(xt)
    return predicted, predictedProba ,svc

def gridSearchSVC(xT, yT):
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC
    from sklearn.metrics import classification_report
    
    hyperparameters = { 'C': [0.1, 1, 100, 1000],
    
    'gamma': [0.0001, 0.001, 0.005, 0.1, 1, 3, 5],
    
    'kernel': ('linear', 'rbf')}
    
    grid = GridSearchCV(estimator=SVC(),param_grid=hyperparameters,cv=5,scoring='accuracy',n_jobs=-1)
    
    grid.fit(xT, yT)
    
    print(f'Best parameters: {grid.best_params_}')
    
    print(f'Best score: {grid.best_score_}')
    
    print(f'All results: {grid.cv_results_}')
    
def knn(xT,yT,xt):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score
    import matplotlib.pyplot as plt
    from sklearn.metrics import accuracy_score
    import numpy as np

    k_range = range(1, 31)
    k_error = []
    for k in k_range:

        knn = KNeighborsClassifier(n_neighbors=k)
        
        #cv parameters determine the proportion of data set partitioning， 5:1split training and test set
        scores = cross_val_score(knn, xT.drop(columns=['X12']), yT, cv=10, scoring='accuracy')
        k_error.append(scores.mean())
        
    plt.plot(k_range, k_error,color='g')

    my_x_ticks = np.arange(0, 31, 1)
    plt.xticks(my_x_ticks,rotation=90)
    
    plt.xlabel('Value of K for KNN')
    
    plt.ylabel('Error')
    
    plt.show()
    
    xT, xt = PCA(xT, xt)
    
    knn = KNeighborsClassifier(n_neighbors=12)

    knn.fit(xT,yT)
    
    predT = knn.predict(xT)
    realT = yT
    trainAccuracy = accuracy_score(predT, realT)
    print("Accuracy: %.2f%%" % (trainAccuracy * 100.0))
    
    predicted = knn.predict(xt)
    predictedProba = knn.predict_proba(xt)
    return predicted ,predictedProba, knn

#%%
#Evaluation

def getMetrics(yPred, yt, proba):
    from sklearn.metrics import classification_report
    from sklearn.metrics import roc_curve, roc_auc_score, auc
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt
    cm = confusion_matrix(yt, yPred)
    print(cm)
    plt.figure(figsize = (8,6))
    sns.heatmap(cm, annot=True, fmt='g')
    plt.show()
    print(classification_report(yt, yPred))
    fpr, tpr, threshold = roc_curve(yt, proba, drop_intermediate=False)
    #print(fpr, tpr, threshold)
    auc1 = auc(fpr, tpr)
    print(auc1)
    plt.title('ROC Curve')
    plt.plot(fpr, tpr, color = 'orange', label = 'AUC = %0.2f' % auc1)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show() 

def probaToPred(proba, threshold):
    result = []
    for i in proba:
        #print(i)
        if i[0] >= threshold:
            result.append(0)
        else:
            result.append(1)
    #print(result)
    return result

#%%
def pipeline():
    import matplotlib.pyplot as plt
    import numpy as np
    
    exploration()
    
    columnsName = data.columns.to_list()
    columnsName.remove('Y')
    
    #train test split
    xT, xt, yT, yt = split(columnsName)

    #remove outlier, choose either one
    #xT, yT  = removeOutlier(xT, yT)
    xT, yT = undersample(xT,yT)
    plt.pie([yT[yT == 0].size, yT[yT == 1].size], autopct=lambda p: '{:.0f}'.format(p * sum([yT[yT == 0].size, yT[yT == 1].size]) / 100),
            labels = yT.unique())
    plt.show()
    
    #oversampling for imbalance learning
    xT, yT = oversample(xT,yT)
    plt.pie([yT[yT == 0].size, yT[yT == 1].size], autopct=lambda p: '{:.0f}'.format(p * sum([yT[yT == 0].size, yT[yT == 1].size]) / 100),
            labels = yT.unique())
    plt.show()
    
    #choose a model
    #pred, proba, model = xgb(xT,yT,xt)
    pred, proba, model = randomForest(xT,yT,xt)
    #pred, proba, model = logReg(xT, yT, xt)
    #pred, proba, model = qda(xT, yT, xt)
    #pred, proba ,model = svm(xT, yT, xt)
    #pred, proba,model = knn(xT,yT,xt)
    
    
    
    #trimming prediction threshold, optional
    #pred = probaToPred(proba, decisionThreshold)
    
    #get performance
    #for those without proba
    #getMetrics(pred, yt, pred)
    #with proba
    getMetrics(pred, yt, proba[:,1])
    
    #get result for official test set
    
    #for forest/xgb
    officialPred = probaToPred(model.predict_proba(officialTest),decisionThreshold)
    officialPred = model.predict(officialTest)
    print(officialPred)
    
    #for logReg/qda
    # cutoff = 0.5 
    # officialTestX12 = pd.get_dummies(officialTest['X12'], prefix="X12_", prefix_sep=" ",drop_first = True)
    # dummyTest = pd.concat([officialTest,officialTestX12], axis =1 )
    # officialPred = np.where(model.predict(dummyTest.drop(columns = ['X12','X11','X3'])) > cutoff, 1,0)
    # print(officialPred)
    
    
    #for svm/knn
    # xT, pcaTest = PCA(xT, officialTest)
    # officialPred = model.predict(pcaTest)
    # print(officialPred)
    

pipeline()
