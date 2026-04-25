import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, recall_score, f1_score,
                              classification_report, confusion_matrix)
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────
df = pd.read_csv('disease_dataset.csv')
print(f"Dataset shape : {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"Diseases      : {df['prognosis'].nunique()}\n")

# ─────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────
X = df.drop('prognosis', axis=1)
y = df['prognosis']

le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)
print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}\n")

# ─────────────────────────────────────────
# 3. TRAIN MODELS
# ─────────────────────────────────────────
models = {
    'Random Forest'     : RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Decision Tree'     : DecisionTreeClassifier(max_depth=20, random_state=42),
    'Gradient Boosting' : GradientBoostingClassifier(n_estimators=80, random_state=42),
    'Naive Bayes'       : GaussianNB(),
    'SVM'               : SVC(kernel='rbf', probability=True, random_state=42),
}

results = {}
print("=" * 55)
print(f"{'Model':<22} {'Accuracy':>10} {'Recall':>10} {'F1':>10}")
print("=" * 55)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred, average='weighted')
    f1  = f1_score(y_test, y_pred, average='weighted')
    results[name] = {'model': model, 'acc': acc, 'rec': rec, 'f1': f1, 'pred': y_pred}
    print(f"{name:<22} {acc*100:>9.2f}% {rec*100:>9.2f}% {f1*100:>9.2f}%")

print("=" * 55)
best_name = max(results, key=lambda k: results[k]['acc'])
print(f"\nBest Model: {best_name} — Accuracy: {results[best_name]['acc']*100:.2f}%\n")

# ─────────────────────────────────────────
# 4. CONFUSION MATRIX
# ─────────────────────────────────────────
best   = results[best_name]
y_pred = best['pred']
cm     = confusion_matrix(y_test, y_pred)
labels = le.classes_

from collections import Counter
top20_idx = [i for i, _ in Counter(y_test).most_common(20)]
top20_idx.sort()
cm20   = cm[np.ix_(top20_idx, top20_idx)]
lbls20 = [labels[i][:18] for i in top20_idx]

fig, ax = plt.subplots(figsize=(14, 11))
sns.heatmap(cm20, annot=True, fmt='d', cmap='Blues',
            xticklabels=lbls20, yticklabels=lbls20,
            linewidths=0.3, ax=ax, annot_kws={'size': 8})
ax.set_title(f'Confusion Matrix — {best_name} (Top 20 Diseases)', fontsize=14, fontweight='bold')
ax.set_xlabel('Predicted Label'); ax.set_ylabel('True Label')
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig('figures/fig2_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("Confusion matrix saved.")

# ─────────────────────────────────────────
# 5. FEATURE IMPORTANCE
# ─────────────────────────────────────────
rf = results['Random Forest']['model']
fi = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print(f"\nTop 10 Critical Symptoms:")
for i, (sym, val) in enumerate(fi.head(10).items(), 1):
    print(f"  {i:>2}. {sym:<35} {val*100:.2f}%")

fig, ax = plt.subplots(figsize=(12, 8))
ax.barh(fi.head(25).index[::-1], fi.head(25).values[::-1]*100, color='steelblue')
ax.set_xlabel('Feature Importance (%)')
ax.set_title('Top 25 Critical Symptoms — Random Forest Feature Importance', fontweight='bold')
plt.tight_layout()
plt.savefig('figures/fig3_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nFeature importance plot saved.")

# ─────────────────────────────────────────
# 6. CLASSIFICATION REPORT
# ─────────────────────────────────────────
print("\n" + "=" * 55)
print("Classification Report (Best Model — SVM)")
print("=" * 55)
print(classification_report(y_test, y_pred, target_names=labels))
