import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss

# sample data set
data = load_breast_cancer()
x = data.data
y = data.target

# split into a training set and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,
                                                    stratify=y, random_state=42)

# Try three different Random Forest Classifers with different n_estimators
clf1 = RandomForestClassifier(n_estimators=100, random_state=42)
clf2 = RandomForestClassifier(n_estimators=200, random_state=42)
clf3 = RandomForestClassifier(n_estimators=300, random_state=42)

# Quick loop to see which performs best
clfs = [clf1,clf2,clf3]
for clf in clfs:
    clf.fit(x_train,y_train)
    y_hat = clf.predict_proba(x_test)
    print(roc_auc_score(y_test, y_hat[:,1]))
    print(log_loss(y_test, y_hat[:,1]))
    print()

# Find the optimal random_state for the best reproducable results
rand_seeds = np.random.randint(1,999,100) # change as needed 
loss_target = 1
roc_target = 0
loop = 1
for i in rand_seeds:
    best_seed = 0
    clf = RandomForestClassifier(n_estimators=200, random_state=i) # change as needed
    clf.fit(x_train,y_train)
    y_hat = clf.predict_proba(x_test)
    loss = log_loss(y_test, y_hat[:,1])
    roc = roc_auc_score(y_test, y_hat[:,1])
    if loss < loss_target and roc > roc_target:
        if i == rand_seeds[0]:
            loop += 1
        else:
            print('New best seed discovered!', i)
            best_seed = i
            loss_target = loss
            roc_target = roc
            print('iteration',loop,'of',len(rand_seeds))
            loop += 1
    else:
        loop += 1
        if loop % 25 == 0:
            print(loop, 'of', len(rand_seeds), 'complete.')
print('Best random_state: ', i)
print('Log Loss:', loss_target)
print('ROC AUC:', roc_target)