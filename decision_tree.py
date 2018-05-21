import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import data_extraction
import plot_learning_curve
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

x_train, y_train, x_test, y_test = data_extraction.get_features_labels('heartbeat-sounds','set_a_training','set_a_testing', 'set_a.csv', 1)


clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
 max_depth=3, min_samples_leaf=5)
clf_entropy.fit(x_train, y_train)
y_pred = clf_entropy.predict(x_test)

print "Test Accuracy is ", accuracy_score(y_test,y_pred)*100


title = "Learning Curves "
# Cross validation with 10 iterations, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
plot_learning_curve.plot(clf_entropy, title, x_train, y_train, ylim=(0.1, 1.01), cv=cv, n_jobs=4)
plt.show()

'''
clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=3, min_samples_leaf=5)
clf_gini.fit(x_train, y_train)
y_pred = clf_gini.predict(x_test)
print "Accuracy is ", accuracy_score(y_test,y_pred)*100'''