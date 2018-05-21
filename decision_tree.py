import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
from scipy.signal import decimate
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import data_extraction

x_train, y_train, x_test, y_test = data_extraction.get_features_labels('heartbeat-sounds','set_a_training','set_a_testing', 'set_a.csv')


clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=3, min_samples_leaf=5)
clf_gini.fit(x_train, y_train)
y_pred = clf_gini.predict(x_test)


print "Accuracy is ", accuracy_score(y_test,y_pred)*100

clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
 max_depth=3, min_samples_leaf=5)
clf_entropy.fit(x_train, y_train)
y_pred = clf_entropy.predict(x_test)

print "Accuracy is ", accuracy_score(y_test,y_pred)*100