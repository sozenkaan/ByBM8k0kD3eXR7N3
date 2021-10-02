import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
df = pd.read_csv('C:\\Users\\bilge\\Desktop\\ACME-HappinessSurvey2020.csv',sep=',')
print(df.head())

df.info()

print(df.isnull().sum())

print(df.describe())

df.cov()

df.hist(figsize=(16,20),bins=50,xlabelsize=8,ylabelsize=8)
#plt.show()

df.boxplot()
#plt.show()

from scipy import stats
df1 = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
print(df1.skew())

df1.boxplot()
#plt.show()

from pandas.plotting import scatter_matrix

pd.plotting.scatter_matrix(df1,alpha=0.3,figsize=(14,8),diagonal='kde')
#plt.show()

y= df1.iloc[:,0:1]
X=df1.iloc[:,1:]


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)

from sklearn.preprocessing import StandardScaler
feature_scaler = StandardScaler()
X_train = feature_scaler.fit_transform(X_train)
X_test = feature_scaler.transform(X_test)


import time


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
 # models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier

dict_classifiers = {
    "Logistic Regression": LogisticRegression(),
    "Nearest Neighbors": KNeighborsClassifier(),
    "Linear SVM": SVC(),
    "Gradient Boosting Classifier": GradientBoostingClassifier(),
    "Decision Tree": tree.DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=18),
    "Neural Net": MLPClassifier(alpha=1),
    "Naive Bayes": GaussianNB(),
    "Extra Trees Classifier": ExtraTreesClassifier(n_estimators=18)
}
from sklearn.model_selection import cross_val_score

# Logistic Regression
log_reg = LogisticRegression()
log_scores = cross_val_score(log_reg, X_train, y_train, cv=10)
log_reg_mean = log_scores.mean()

# SVC
svc_clf = SVC()
svc_scores = cross_val_score(svc_clf, X_train, y_train, cv=10)
svc_mean = svc_scores.mean()

# KNearestNeighbors
knn_clf = KNeighborsClassifier()
knn_scores = cross_val_score(knn_clf, X_train, y_train, cv=10)
knn_mean = knn_scores.mean()

# Decision Tree
tree_clf = tree.DecisionTreeClassifier()
tree_scores = cross_val_score(tree_clf, X_train, y_train, cv=10)
tree_mean = tree_scores.mean()

# Gradient Boosting Classifier
grad_clf = GradientBoostingClassifier()
grad_scores = cross_val_score(grad_clf, X_train, y_train, cv=10)
grad_mean = grad_scores.mean()

# Random Forest Classifier
rand_clf = RandomForestClassifier(n_estimators=10)
rand_scores = cross_val_score(rand_clf, X_train, y_train, cv=10)
rand_mean = rand_scores.mean()

# NeuralNet Classifier
neural_clf = MLPClassifier(alpha=1)
neural_scores = cross_val_score(neural_clf, X_train, y_train, cv=10)
neural_mean = neural_scores.mean()

# Naives Bayes
nav_clf = GaussianNB()
nav_scores = cross_val_score(nav_clf, X_train, y_train, cv=10)
nav_mean = neural_scores.mean()

 #ExtraTreesClassifier
ext_clf = ExtraTreesClassifier(n_estimators=10)
ext_scores = cross_val_score(ext_clf, X_train, y_train, cv=10)
ext_mean = ext_scores.mean()



# Create a Dataframe with the results.
d = {'Classifiers': ['Logistic Reg.', 'SVC', 'KNN', 'Dec Tree', 'Grad B CLF', 'Rand FC', 'Neural Classifier', 'Naives Bayes','Ext_clf'],
    'Crossval Mean Scores': [log_reg_mean, svc_mean, knn_mean, tree_mean, grad_mean, rand_mean, neural_mean, nav_mean, ext_mean]}

result_df = pd.DataFrame(data=d)

result_df = result_df.sort_values(by=['Crossval Mean Scores'], ascending=False)
print(result_df)
from sklearn.model_selection import cross_val_predict

y_test_pred_1 = cross_val_predict(grad_clf, X_test, y_test, cv=10)
y_test_pred_2 = cross_val_predict(log_reg, X_test, y_test, cv=10)
y_test_pred_3 = cross_val_predict(svc_clf, X_test, y_test, cv=10)
y_test_pred_4 = cross_val_predict(knn_clf, X_test, y_test, cv=10)
y_test_pred_5 = cross_val_predict(tree_clf, X_test, y_test, cv=10)
y_test_pred_6 = cross_val_predict(rand_clf, X_test, y_test, cv=10)
y_test_pred_7 = cross_val_predict(neural_clf, X_test, y_test, cv=10)
y_test_pred_8 = cross_val_predict(nav_clf, X_test, y_test, cv=10)
y_test_pred_9 = cross_val_predict(ext_clf, X_test, y_test, cv=10)

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

print("Gradient Boosting Classifier -------------------------------------------------")
print('Accuracy Score: ', accuracy_score(y_test, y_test_pred_1))
print('Precision Score: ', precision_score(y_test, y_test_pred_1))
print('Recall Score: ', recall_score(y_test, y_test_pred_1))
print('F1 Score: ', f1_score(y_test, y_test_pred_1))
print("Logistic Regression -------------------------------------------------")
print('Accuracy Score: ', accuracy_score(y_test, y_test_pred_2))
print('Precision Score: ', precision_score(y_test, y_test_pred_2))
print('Recall Score: ', recall_score(y_test, y_test_pred_2))
print('F1 Score: ', f1_score(y_test, y_test_pred_2))
print("SVM -------------------------------------------------")
print('Accuracy Score: ', accuracy_score(y_test, y_test_pred_3))
print('Precision Score: ', precision_score(y_test, y_test_pred_3))
print('Recall Score: ', recall_score(y_test, y_test_pred_3))
print('F1 Score: ', f1_score(y_test, y_test_pred_3))
print("KNeighbors Classifier-------------------------------------------------")
print('Accuracy Score: ', accuracy_score(y_test, y_test_pred_4))
print('Precision Score: ', precision_score(y_test, y_test_pred_4))
print('Recall Score: ', recall_score(y_test, y_test_pred_4))
print('F1 Score: ', f1_score(y_test, y_test_pred_4))
print("Decision Trees Classifier-------------------------------------------------")
print('Accuracy Score: ', accuracy_score(y_test, y_test_pred_5))
print('Precision Score: ', precision_score(y_test, y_test_pred_5))
print('Recall Score: ', recall_score(y_test, y_test_pred_5))
print('F1 Score: ', f1_score(y_test, y_test_pred_5))
print("Random Forest Classifier-------------------------------------------------")
print('Accuracy Score: ', accuracy_score(y_test, y_test_pred_6))
print('Precision Score: ', precision_score(y_test, y_test_pred_6))
print('Recall Score: ', recall_score(y_test, y_test_pred_6))
print('F1 Score: ', f1_score(y_test, y_test_pred_6))
print("MLP Classifier-------------------------------------------------")
print('Accuracy Score: ', accuracy_score(y_test, y_test_pred_7))
print('Precision Score: ', precision_score(y_test, y_test_pred_7))
print('Recall Score: ', recall_score(y_test, y_test_pred_7))
print('F1 Score: ', f1_score(y_test, y_test_pred_7))
print("GaussianNB-------------------------------------------------")
print('Accuracy Score: ', accuracy_score(y_test, y_test_pred_8))
print('Precision Score: ', precision_score(y_test, y_test_pred_8))
print('Recall Score: ', recall_score(y_test, y_test_pred_8))
print('F1 Score: ', f1_score(y_test, y_test_pred_8))
print("Extra Trees Classifier -------------------------------------------------")
print('Accuracy Score: ', accuracy_score(y_test, y_test_pred_9))
print('Precision Score: ', precision_score(y_test, y_test_pred_9))
print('Recall Score: ', recall_score(y_test, y_test_pred_9))
print('F1 Score: ', f1_score(y_test, y_test_pred_9))

predicted_y = cross_val_predict(rand_clf, X, y, cv=10)
print(predicted_y)



from sklearn.feature_selection import SelectKBest, chi2
X_5_best= SelectKBest(chi2, k=5).fit(X, y)
mask = X_5_best.get_support() #list of booleans for selected features
new_feat = []
for bool, feature in zip(mask, X.columns):
    if bool:
        new_feat.append(feature)
print("The best features are:{}".format(new_feat))

