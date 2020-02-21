import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator , TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Datenbeschaffung / Einlesen einer csv Datei wie in Abschnitt 2 beschrieben
def load_housing_data():
    csv_path = os.path.join("datasets/housing/housing.csv")
    return pd.read_csv(csv_path)

housing = load_housing_data()
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self # nothing else to do

    # Da die Methode nur die Werte übergeben bekommt. muss auf das Array mit Spaltennummern (rooms_ix etc.) zugegriffen werden
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

# Erstellung income category Attribut mit fünf Kategorien
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

# Basierend auf dem Kategorie-Attribut wird nun eine stratifizierte Stichprobe gezogen
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

housing = strat_train_set.drop("median_house_value",axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# Erstellen eines Dataframes ohne kategorielle Attribute
housing_num = housing.drop("ocean_proximity", axis=1)

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

# Klasse für die Auswahl nummerischer und kategorieller Spalten
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

# Pipeline für die Verarbeitung nummerischer Attribute
num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
    ])

# Pipeline für die Verarbeitung kategorieller Attribute
cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('label_binarizer', OneHotEncoder()),
    ])

# Zusammensetzen der Teil-Pipelines
full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline),
    ])

# Bis hierher arbeitet die pipeline noch nicht mit echten Daten. Sie verfügt nur über das Wissen über die Attribute und der
# Transformationsfunktionen. Erst jetzt werden der Pipeline echte housing-Daten übergeben:
housing_prepared = full_pipeline.fit_transform(housing)

# Erstellung eines SVM Modells

def execute_SVM():
    param_grid = [
        {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
        {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
    ]

    svm_reg = RandomForestRegressor()
    random_search = RandomizedSearchCV(svm_reg, param_distributions=param_grid, cv=5, scoring='neg_mean_squared_error')
    random_search.fit(housing_prepared, housing_labels)

    # Ausgabe der besten Paramter:
    print(random_search.best_params_)

    print(random_search.best_estimator_)

    # Alle Scores der Leistungsmessungen ausgeben
    cvres = random_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)


execute_SVM()

