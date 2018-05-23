import os
import sys
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

directory = 'set_'+sys.argv[1]
wavelet_transform = int(sys.argv[2])

x_train, y_train, x_test, y_test = data_extraction.get_features_labels(directory, wavelet_transform)


clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
 max_depth=3, min_samples_leaf=5)
clf_entropy.fit(x_train, y_train)
y_pred = clf_entropy.predict(x_test)

print("Entropy Test Accuracy is "+str(accuracy_score(y_test,y_pred)*100))


title = "Learning Curves Entropy"
# Cross validation with 10 iterations, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
plot_learning_curve.plot(clf_entropy, title, x_train, y_train, ylim=(0.1, 1.01), cv=cv, n_jobs=4)
plt.show()

clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=3, min_samples_leaf=5)
clf_gini.fit(x_train, y_train)
y_pred = clf_gini.predict(x_test)
print("Gini Test Accuracy is ", accuracy_score(y_test,y_pred)*100)

title = "Learning Curves Gini"
# Cross validation with 10 iterations, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
plot_learning_curve.plot(clf_entropy, title, x_train, y_train, ylim=(0.1, 1.01), cv=cv, n_jobs=4)
plt.show()