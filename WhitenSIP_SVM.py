import pandas as pd
import deepchem as dc
import numpy as np
import sklearn
from sklearn.model_selection import GridSearchCV
def seed_all():
    np.random.seed(123)


# dataset
df = pd.read_csv('Data\WhitenSIP.csv', encoding='gbk')
df = df[['canonical_smiles','y']]
df.columns=['smiles','label']

#split
X_train,X_test,y_train,y_test = sklearn.model_selection.train_test_split(df['smiles'],df['label'],test_size=0.2,random_state=24)

#featurizer
featurizer = dc.feat.MACCSKeysFingerprint()
train_features = featurizer.featurize(X_train)
test_features = featurizer.featurize(X_test)

#model
param_grid = {'penalty':['l1','l2'],
             'C':np.linspace(0.1,1,10),
             'solver':['liblinear']}

logreg = sklearn.linear_model.LogisticRegression()
grid_model = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=5,scoring='roc_auc')
grid_model_result = grid_model.fit(train_features, y_train)

# Summarize results
best_score, best_params = grid_model_result.best_score_, grid_model_result.best_params_
print("Best: %f using %s" % (best_score, best_params))

# Extract the best model and evaluate it on the test set
best_model = grid_model_result.best_estimator_
print("%s of LogisticRegression: %f"%('roc_auc',best_model.score(test_features, y_test)))

logreg.fit(train_features,y_train)