#!/usr/bin/python

import sys
import pickle
#sys.path.append("../tools/")
sys.path.append("C:/Users/geoffnoble/Source/Repos/ud120-projects/tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

#for some reason I can't get my folder references right in Visual Studio
path_name = "C:/Users/geoffnoble/Source/Repos/ud120-projects/final_project/"

#add all feautures before testing, removed email address
features_list = ['poi','salary', 'deferral_payments', 
                 'total_payments', 'loan_advances', 'bonus',
                 'restricted_stock_deferred', 'deferred_income',
                 'total_stock_value', 'expenses', 'exercised_stock_options',
                 'other', 'long_term_incentive', 'restricted_stock', 
                 'director_fees', 'to_messages', 'from_poi_to_this_person', 'from_messages', 
                 'from_this_person_to_poi', 'shared_receipt_with_poi' ] 

### Load the dictionary containing the dataset
with open(path_name + "final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop("TOTAL", 0)



### Task 3: Create new feature(s)
def dict_to_list(key,normalizer):
    new_list=[]

    for i in data_dict:
        if data_dict[i][key]=="NaN" or data_dict[i][normalizer]=="NaN":
            new_list.append(0.)
        elif data_dict[i][key]>=0:
            new_list.append(float(data_dict[i][key])/float(data_dict[i][normalizer]))
    return new_list

### create two lists of new features
fraction_from_poi_email=dict_to_list("from_poi_to_this_person","to_messages")
fraction_to_poi_email=dict_to_list("from_this_person_to_poi","from_messages")

### insert new features into data_dict
count=0
for i in data_dict:
    data_dict[i]["fraction_from_poi_email"]=fraction_from_poi_email[count]
    data_dict[i]["fraction_to_poi_email"]=fraction_to_poi_email[count]
    count +=1


### Store to my_dataset for easy export below.
my_dataset = data_dict

### Go back to feature selection
data = featureFormat(my_dataset, features_list, sort_keys = True)

labels, features = targetFeatureSplit(data)

#Select labels using SelectKBest after preprocessing and scaling
scaler = preprocessing.StandardScaler()
kbest = SelectKBest(f_classif, k=10)

pipeline =  Pipeline(steps=[("kbest", kbest)])

pipelineSCALED =  Pipeline(steps=[('scaling', scaler),("kbest", kbest)])

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

pipeline.fit(features_train, labels_train)
pipelineSCALED.fit(features_train, labels_train)

kbest.fit(features, labels)
features = kbest.transform(features)
feature_scores = zip(features_list[1:],kbest.scores_)

feature_list_to_show = sorted(feature_scores, key=lambda feature: feature[1], reverse = True)
for item in feature_list_to_show:
 	print item[0], item[1]

## Update features_list based on DecisionTreeClassifier importances
features_list = ['poi','salary',  
                 'total_payments', 'bonus',
                 'restricted_stock_deferred', 'deferred_income',
                 'total_stock_value', 'exercised_stock_options',
                 'long_term_incentive', 'restricted_stock', 
                 'loan_advances', 'expenses', 'from_poi_to_this_person']                 

### Task 4: Try a varity of classifiers
#GaussianNB
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(features_train,labels_train)
score = clf.score(features_test,labels_test)
print "GaussianNB : " + str(score)

from tester import test_classifier
test_classifier(clf, my_dataset, features_list)

#AdaBoost
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(n_estimators=50, random_state=42)
clf.fit(features_train,labels_train)
score = clf.score(features_test,labels_test)
print "AdaBoost : " + str(score)


### SVM, commented out as slow
#from sklearn.svm import SVC
#clf = SVC(C=5000, gamma=0.0001, kernel='linear', random_state=42)
#clf.fit(features_train,labels_train)
#score = clf.score(features_test,labels_test)
#print "SVM : " + str(score)




### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

from sklearn.svm import SVC
from sklearn import svm, grid_search, datasets

#clf = SVC(kernel="linear", C=1.)
#clf.fit(features_train, labels_train)

#print clf.score(features_test, labels_test)

parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svr = svm.SVC()
clf = grid_search.GridSearchCV(svr, parameters)
clf.fit(iris.data, iris.target)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)