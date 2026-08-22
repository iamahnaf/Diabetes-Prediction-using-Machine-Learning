import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix, accuracy_score
import pickle

np.random.seed(42)

# Load and preprocess
df = pd.read_csv('uiu100k.csv')
le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])
df['smoking_history'] = le.fit_transform(df['smoking_history'])
X = df.drop('diabetes', axis=1).values
y = df['diabetes'].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full)

results = []

# helper: find best threshold meeting recall target maximizing precision

def tune_threshold(probas, y_val, target_recall=0.85):
    best = {'threshold': 0.5, 'precision': -1, 'recall': -1, 'f1': -1}
    for t in np.linspace(0.1, 0.9, 161):
        pred = (probas >= t).astype(int)
        r = recall_score(y_val, pred)
        p = precision_score(y_val, pred, zero_division=0)
        f = f1_score(y_val, pred, zero_division=0)
        if r >= target_recall and p > best['precision']:
            best.update({'threshold': float(t), 'precision': float(p), 'recall': float(r), 'f1': float(f)})
    # fallback: highest recall
    if best['precision'] < 0:
        for t in np.linspace(0.1,0.9,161):
            pred = (probas >= t).astype(int)
            r = recall_score(y_val, pred)
            p = precision_score(y_val, pred, zero_division=0)
            f = f1_score(y_val, pred, zero_division=0)
            if r > best['recall']:
                best.update({'threshold': float(t), 'precision': float(p), 'recall': float(r), 'f1': float(f)})
    return best

# Try LightGBM
lgb_available = True
try:
    import lightgbm as lgb
except Exception as e:
    lgb_available = False

if lgb_available:
    print('Training LightGBM...')
    model = lgb.LGBMClassifier(n_estimators=500, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    proba_val = model.predict_proba(X_val)[:,1]
    best = tune_threshold(proba_val, y_val)
    proba_test = model.predict_proba(X_test)[:,1]
    pred_test = (proba_test >= best['threshold']).astype(int)
    p = precision_score(y_test, pred_test, zero_division=0)
    r = recall_score(y_test, pred_test)
    f = f1_score(y_test, pred_test, zero_division=0)
    cm = confusion_matrix(y_test, pred_test)
    results.append(('LightGBM', model, best, {'precision':p,'recall':r,'f1':f,'cm':cm}))
    print('LightGBM done. Val best:', best, 'Test metrics precision/recall/f1', (p,r,f))
else:
    print('LightGBM not available.')

# Try XGBoost
xgb_available = True
try:
    import xgboost as xgb
except Exception as e:
    xgb_available = False

if xgb_available:
    print('Training XGBoost...')
    # compute scale_pos_weight for imbalance
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    scale_pos_weight = float(n_neg) / max(1.0, float(n_pos))
    model = xgb.XGBClassifier(n_estimators=500, use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos_weight, random_state=42)
    model.fit(X_train, y_train)
    proba_val = model.predict_proba(X_val)[:,1]
    best = tune_threshold(proba_val, y_val)
    proba_test = model.predict_proba(X_test)[:,1]
    pred_test = (proba_test >= best['threshold']).astype(int)
    p = precision_score(y_test, pred_test, zero_division=0)
    r = recall_score(y_test, pred_test)
    f = f1_score(y_test, pred_test, zero_division=0)
    cm = confusion_matrix(y_test, pred_test)
    results.append(('XGBoost', model, best, {'precision':p,'recall':r,'f1':f,'cm':cm}))
    print('XGBoost done. Val best:', best, 'Test metrics precision/recall/f1', (p,r,f))
else:
    print('XGBoost not available.')

# Fallback: sklearn's HistGradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
print('Training HistGradientBoostingClassifier (sklearn)...')
model = HistGradientBoostingClassifier(max_iter=500, random_state=42)
model.fit(X_train, y_train)
proba_val = model.predict_proba(X_val)[:,1]
best = tune_threshold(proba_val, y_val)
proba_test = model.predict_proba(X_test)[:,1]
pred_test = (proba_test >= best['threshold']).astype(int)
p = precision_score(y_test, pred_test, zero_division=0)
r = recall_score(y_test, pred_test)
f = f1_score(y_test, pred_test, zero_division=0)
cm = confusion_matrix(y_test, pred_test)
results.append(('HistGB', model, best, {'precision':p,'recall':r,'f1':f,'cm':cm}))
print('HistGB done. Val best:', best, 'Test metrics precision/recall/f1', (p,r,f))

# Summarize and save best by precision (with recall>=0.85)
best_overall = None
for name, mdl, bestth, testm in results:
    if bestth['recall'] >= 0.85:
        if best_overall is None or testm['precision'] > best_overall[3]['precision']:
            best_overall = (name, mdl, bestth, testm)

if best_overall is None:
    # choose highest recall
    for name, mdl, bestth, testm in results:
        if best_overall is None or testm['recall'] > best_overall[3]['recall']:
            best_overall = (name, mdl, bestth, testm)

if best_overall is not None:
    name, mdl, bestth, testm = best_overall
    print('Best overall model chosen:', name)
    print('Validation threshold:', bestth)
    print('Test metrics:', testm)
    # save model and scaler
    with open('best_gbm_model.pkl', 'wb') as f:
        pickle.dump({'model_name': name, 'model': mdl, 'threshold': bestth['threshold']}, f)
    print('Saved best model to best_gbm_model.pkl')
else:
    print('No models produced results')

# print a short recommendation
print('\nRecommendation:')
print('- Use the saved model best_gbm_model.pkl for inference; it includes the chosen threshold.')
print('- To integrate into the notebook, load the model and use model.predict_proba(X)[:,1] >= threshold')
