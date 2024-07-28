# Standard Library Imports
import joblib

# Third-Party Imports
import pandas as pd
import datetime as dt
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from imblearn.over_sampling import SMOTE

# Django Imports
from django.core.cache import caches


def model_training(data):
    with open(caches['cloud'].get('scaling'), 'rb') as f:
        scaling = joblib.load(f)

    X = data.iloc[:, :-2]
    X = scaling.transform(X)
    Y = data.iloc[:, -1]
    Y_awake = Y[:].apply(lambda x: "AWAKE" if x == "AWAKE" else "ASLEEP")

    seed = 7
    step1 = BaggingClassifier(RandomForestClassifier(class_weight='balanced'), random_state=seed)
    step1.fit(X, Y_awake)

    X_asleep = X[Y_awake == 'ASLEEP']
    Y_asleep = Y[Y_awake == 'ASLEEP']

    smote = SMOTE(random_state=seed)
    X_resampled, Y_resampled = smote.fit_resample(X_asleep, Y_asleep)

    step2 = BaggingClassifier(RandomForestClassifier(), random_state=seed)
    step2.fit(X_resampled, Y_resampled)

    models = {'Step 1': step1, 'Step 2': step2}
    return models


def ML_predict(data, models, scaling):
    try:
        time_data = [dt.datetime.strptime(str(i), '%Y-%m-%d %H:%M:%S') for i in data.values[:, -1]]

        # Step 1
        predictions = pd.DataFrame(index=data.index, columns=['score'])
        clean_indicies = data.dropna().index

        predictions_clean = pd.DataFrame(models['Step 1'].predict(scaling.transform(data.dropna().iloc[:, :-1])))
        predictions.loc[clean_indicies] = predictions_clean.values

        # Step 2
        if 'ASLEEP' in predictions['score'].values:
            asleep_indices = predictions[predictions['score'] == 'ASLEEP'].index
            data_asleep = data.loc[asleep_indices, :]

            predictions_asleep = pd.DataFrame(models['Step 2'].predict(scaling.transform(data_asleep.iloc[:, :-1])))
            predictions.loc[asleep_indices] = predictions_asleep.values

        scored_data = pd.DataFrame({'time_stamp': time_data, 'score': predictions['score']}, dtype=str)
        return scored_data

    except Exception as e:
        # Log error and return None or raise exception
        return None
