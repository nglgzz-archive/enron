#!/usr/bin/python
from tester import dump_classifier_and_data
from pprint import PrettyPrinter
import pickle, sys

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

# returns the string representation containing just 3 decimals
float_format = lambda x: "{0:.3f}".format(x)
# used for pretty printing
pp = PrettyPrinter(indent=4)



### Task 1: Select what feature you'll use
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
print "====================================================="
print "=      Task 1: Select what features you'll use      ="
print "=====================================================\n"

# I thought of using features that best represent importance and contacts
# of a person to classify persons of interest. For email features I'll be 
# using the ones I'll create, while for financial features I'll keep
# some of the existing ones.   
features_list = ["poi", "salary", "bonus", "total_stock_value",
				 "from_poi_perc", "to_poi_perc"]

# Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
	data_dict = pickle.load(data_file)


# counts how many NaN values there are for a given feature
count_nan = lambda f: len([1 for name in data_dict if data_dict[name][f] == "NaN"])
n_poi = len([1 for name in data_dict if data_dict[name]["poi"]])

print "Employees count:", len(data_dict)
print "Poi count:", n_poi
print "NaN count:"
pp.pprint({f:count_nan(f) for f in data_dict["TOTAL"]})

print "\nChosen features:"
pp.pprint(features_list)
print "\n"


### Task 2: Remove outliers
print "====================================================="
print "=              Task 2: Remove outliers              ="
print "=====================================================\n"

# All financial features have an outlier which is sum of all the other values,
# this can be also seen on the insiderpay pdf.
from collections import defaultdict
outlier_values = defaultdict(lambda:0)
outlier_names = {}
negative_names = set([])

# Features that I'll be using and contain said outlier.
outlier_features = ["salary", "bonus", "total_stock_value"]

# Find the maximum value for the features in the outlier_features list,
# save their value and key (name of the person). I expect the keys being
# something like total, and the values the same values as in the total
# row in the insiderpay pdf.
for key in data_dict:
	for feature in outlier_features:
		if data_dict[key][feature] != "NaN":
			if data_dict[key][feature] > outlier_values[feature]:
				outlier_values[feature] = data_dict[key][feature]
				outlier_names[feature] = key
			# When trying the pipeline converting the NaN values to 0 instead 
			# of -1 chi2 throws an error saying there's
			# a negative value, so here I check for that value 
			if data_dict[key][feature] < 0:
				negative_names.add(key)
		else:
			data_dict[key][feature] = -1

print "Values of the outliers:"
# I converted the defaultdict to dict in order for pprint to format properly.
pp.pprint(dict(outlier_values))
print "Name containing the outliers:"
pp.pprint(outlier_names)
print "Name containing negative values:"
print negative_names
pp.pprint([data_dict[name] for name in negative_names])
print "\n"

# As expected the key containing all the outliers was TOTAL,
# so I'll just remove this key from the datset.
data_dict.pop("TOTAL")
# The name containing the negative value was BELFER ROBERT, and it had
# a negative stock value which doesn't make sense, since stocks can't
# be negative.
data_dict.pop("BELFER ROBERT")



### Task 3: Create new feature(s)
print "====================================================="
print "=           Task 3: Create new feature(s)           ="
print "=====================================================\n"

from math import isnan
perc = lambda x,y: float(x)/float(y)

for key in data_dict:
	current = data_dict[key]
	from_poi = perc(current["from_poi_to_this_person"], current["to_messages"])
	to_poi = perc(current["from_this_person_to_poi"], current["from_messages"])
	data_dict[key]["from_poi_perc"] = -1 if isnan(from_poi) else from_poi
	data_dict[key]["to_poi_perc"] = -1 if isnan(to_poi) else to_poi	

print "Created:"
print "\tfrom_poi_perc - percentage of emails received from poi"
print "\tto_poi_perc - percentage of emails sent to poi\n\n"

# Store to my_dataset for easy export below.
my_dataset = data_dict

# Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)



### Task 4: Try a variety of classifiers
### Please name your classifier "clf" for easy export below.
### Note that id you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
print "====================================================="
print "=       Task 4: Try a variety of classifiers        ="
print "=====================================================\n"

# Function to test a classifier, uses StratifiedShuffle split
# method to split train and test data
def test_classifier(clf, features, labels, n_iter = 1000):
	from sklearn.cross_validation import StratifiedShuffleSplit
	sss = StratifiedShuffleSplit(labels, n_iter = n_iter, test_size = 0.1, 
		random_state = 42)

	from sklearn.metrics import accuracy_score
	from sklearn.metrics import precision_score
	from sklearn.metrics import recall_score
	from sklearn.metrics import f1_score
	import numpy as np

	features = np.array(features)
	labels = np.array(labels)

	accuracy_list = []
	precision_list = []
	recall_list = []
	f1_list = []

	for train_index, test_index in sss:
		# Extract the train and test data from the given indexes
		features_train = features[train_index]
		features_test = features[test_index]
		labels_train = labels[train_index]
		labels_test = labels[test_index]

		# Fit the classifier and then make a prediction with the test data
		clf.fit(features_train, labels_train)
		pred = clf.predict(features_test)

		# Append the scores in the lists for later calculations 
		accuracy_list.append(accuracy_score(labels_test, pred))
		precision_list.append(precision_score(labels_test, pred))
		recall_list.append(recall_score(labels_test, pred))
		f1_list.append(f1_score(labels_test, pred))

	mean = lambda x: float_format(sum(x)/float(len(x)))
	print clf
	print "Accuracy: ", mean(accuracy_list)
	print "Precision: ", mean(precision_list), " | Recall: ", mean(recall_list)
	print "F1: ", mean(f1_list)
	print ""

from sklearn.svm import SVC
clf_svc = SVC()
test_classifier(clf_svc, features, labels, 100)

from sklearn.neighbors import KNeighborsClassifier
clf_knc = KNeighborsClassifier(n_neighbors = 5)
test_classifier(clf_knc, features, labels, 100)

from sklearn.naive_bayes import GaussianNB
clf_gnb = GaussianNB()
test_classifier(clf_gnb, features, labels, 100)

from sklearn.tree import DecisionTreeClassifier
clf_dtc = DecisionTreeClassifier(min_samples_split = 100, class_weight = "balanced")
test_classifier(clf_dtc, features, labels, 100)
print ""



### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratifies shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
print "====================================================="
print "=           Task 5: Tune your classifier            ="
print "=====================================================\n"

from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

scaler = MinMaxScaler()
pca = PCA(n_components = "mle")

selectors = []
selectors.append(SelectKBest(chi2, k = "all"))
selectors.append(SelectKBest(chi2, k = 4))
selectors.append(SelectKBest(chi2, k = 3))

trees = []
trees.append(DecisionTreeClassifier(min_samples_split = 100, class_weight = None))
trees.append(DecisionTreeClassifier(min_samples_split = 25, class_weight = "balanced"))
trees.append(DecisionTreeClassifier(min_samples_split = 50, class_weight = "balanced"))
trees.append(DecisionTreeClassifier(min_samples_split = 100, class_weight = "balanced"))

print "Test with different Ks in selector:"
for selector in selectors:
	pipeline = [("scaler", scaler), ("selector", selector), ("pca", pca), ("tree", trees[1])]
	clf = Pipeline(pipeline)
	test_classifier(clf, features, labels, 100)	

print "\nTest with different tree parameters:"
for tree in trees:
	pipeline = [("scaler", scaler), ("selector", selectors[0]), ("pca", pca), ("tree", tree)]
	clf = Pipeline(pipeline)
	test_classifier(clf, features, labels, 100)


print "Chosen combination:"
# The best parameters turned out to be min_samples_split = 25, class_weight = "balanced"
pipeline = [("scaler", scaler), ("selector", selectors[0]), ("pca", pca), ("tree", trees[1])]
clf = Pipeline(pipeline)
test_classifier(clf, features, labels)

print "P-values:"
# Print the p-values of the features 
pp.pprint(zip(features_list[1:], map(float_format, clf.named_steps["selector"].pvalues_)))
print "\n"



### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results
print "====================================================="
print "=               Task 6: Dump your data              ="
print "=====================================================\n"

dump_classifier_and_data(clf, my_dataset, features_list)
print "Done.\n"
