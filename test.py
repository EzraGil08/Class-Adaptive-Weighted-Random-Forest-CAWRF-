import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix
from sklearn.datasets import fetch_openml
from sklearn.datasets import get_data_home
import shutil


class CAWRF:
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees = []
        self.classes_ = None
        self.n_classes_ = None
        self.weight_matrix = None
        
    def fit(self, X_train, y_train, X_val, y_val):
        self.classes_ = np.unique(y_train)
        self.n_classes_ = len(self.classes_)
        n_samples = len(X_train)
        
        for i in range(self.n_estimators):
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            tree = DecisionTreeClassifier(max_depth=self.max_depth, random_state=self.random_state + i)
            tree.fit(X_train[indices], y_train[indices])
            self.trees.append(tree)
        
        self._compute_weights(X_val, y_val)
        return self
    
    def _compute_weights(self, X_val, y_val):
        self.weight_matrix = np.zeros((self.n_estimators, self.n_classes_))
        
        for t, tree in enumerate(self.trees):
            y_pred = tree.predict(X_val)
            for c_idx, c in enumerate(self.classes_):
                class_mask = (y_val == c)
                n_class = np.sum(class_mask)
                if n_class > 0:
                    correct = np.sum((y_pred == c) & class_mask)
                    self.weight_matrix[t, c_idx] = correct / n_class
    
    def predict_proba(self, X):
        weighted_votes = np.zeros((len(X), self.n_classes_))
        for t, tree in enumerate(self.trees):
            proba = tree.predict_proba(X)
            for c in range(self.n_classes_):
                weighted_votes[:, c] += proba[:, c] * self.weight_matrix[t, c]
        sums = weighted_votes.sum(axis=1, keepdims=True)
        sums[sums == 0] = 1
        return weighted_votes / sums
    
    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class OOB_WeightedRF:
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees = []
        self.tree_weights = None
        self.classes_ = None
        self.oob_indices = []
        
    def fit(self, X_train, y_train):
        self.classes_ = np.unique(y_train)
        n_samples = len(X_train)
        
        for i in range(self.n_estimators):
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            oob = np.setdiff1d(np.arange(n_samples), indices)
            self.oob_indices.append(oob)
            
            tree = DecisionTreeClassifier(max_depth=self.max_depth, random_state=self.random_state + i)
            tree.fit(X_train[indices], y_train[indices])
            self.trees.append(tree)
        
        self._compute_oob_weights(X_train, y_train)
        return self
    
    def _compute_oob_weights(self, X_train, y_train):
        self.tree_weights = np.zeros(self.n_estimators)
        
        for t, tree in enumerate(self.trees):
            oob = self.oob_indices[t]
            if len(oob) > 0:
                oob_pred = tree.predict(X_train[oob])
                self.tree_weights[t] = accuracy_score(y_train[oob], oob_pred)
        
        if self.tree_weights.sum() > 0:
            self.tree_weights = self.tree_weights / self.tree_weights.sum()
    
    def predict_proba(self, X):
        weighted_votes = np.zeros((len(X), len(self.classes_)))
        for t, tree in enumerate(self.trees):
            proba = tree.predict_proba(X)
            weighted_votes += proba * self.tree_weights[t]
        return weighted_votes
    
    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# DATA

#letters = fetch_openml('letter', parser='auto')
#data = pd.DataFrame(letters.data)
#data['target'] = letters.target
#data.to_csv('data.csv', index=False)

data = pd.read_csv('discdiabetes.csv')
#data = resample(data, n_samples=50000, random_state=42, stratify=data['Class'])


X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

rf = RandomForestClassifier(n_estimators=75, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

oob_rf = OOB_WeightedRF(n_estimators=75, random_state=42)
oob_rf.fit(X_train, y_train)
oob_rf_pred = oob_rf.predict(X_test)

cawrf = CAWRF(n_estimators=75, random_state=42)
cawrf.fit(X_train, y_train, X_val, y_val)
cawrf_pred = cawrf.predict(X_test)

rf_results = {
    'accuracy': accuracy_score(y_test, rf_pred),
    'precision': precision_recall_fscore_support(y_test, rf_pred, average=None)[0],
    'recall': precision_recall_fscore_support(y_test, rf_pred, average=None)[1],
    'f1': precision_recall_fscore_support(y_test, rf_pred, average=None)[2],
    'macro_f1': precision_recall_fscore_support(y_test, rf_pred, average='macro')[2]
}

oob_rf_results = {
    'accuracy': accuracy_score(y_test, oob_rf_pred),
    'precision': precision_recall_fscore_support(y_test, oob_rf_pred, average=None)[0],
    'recall': precision_recall_fscore_support(y_test, oob_rf_pred, average=None)[1],
    'f1': precision_recall_fscore_support(y_test, oob_rf_pred, average=None)[2],
    'macro_f1': precision_recall_fscore_support(y_test, oob_rf_pred, average='macro')[2]
}

cawrf_results = {
    'accuracy': accuracy_score(y_test, cawrf_pred),
    'precision': precision_recall_fscore_support(y_test, cawrf_pred, average=None)[0],
    'recall': precision_recall_fscore_support(y_test, cawrf_pred, average=None)[1],
    'f1': precision_recall_fscore_support(y_test, cawrf_pred, average=None)[2],
    'macro_f1': precision_recall_fscore_support(y_test, cawrf_pred, average='macro')[2]
}

print()
print("RANDOM FOREST")
print(f"Accuracy: {rf_results['accuracy']:.4f}")
print(f"Macro F1: {rf_results['macro_f1']:.4f}")
print(f"Per-Class F1: {rf_results['f1']}")
print(confusion_matrix(y_test, rf_pred))

print("\nOOB-WEIGHTED RF")
print(f"Accuracy: {oob_rf_results['accuracy']:.4f}")
print(f"Macro F1: {oob_rf_results['macro_f1']:.4f}")
print(f"Per-Class F1: {oob_rf_results['f1']}")
print(confusion_matrix(y_test, oob_rf_pred))

print("\nCAWRF")
print(f"Accuracy: {cawrf_results['accuracy']:.4f}")
print(f"Macro F1: {cawrf_results['macro_f1']:.4f}")
print(f"Per-Class F1: {cawrf_results['f1']}")
print(confusion_matrix(y_test, cawrf_pred))
print()