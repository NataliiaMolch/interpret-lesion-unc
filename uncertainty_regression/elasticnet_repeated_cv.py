from sklearn.feature_selection import RFE, SelectKBest, f_regression, VarianceThreshold, mutual_info_regression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import AgglomerativeClustering
from interpret.glassbox import ExplainableBoostingRegressor
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score, f1_score, accuracy_score, make_scorer
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from pprint import pprint
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning



# Global random state range
RNG = np.random.RandomState(0)

# Features selection parameters
FEATSEL_PARAMS = {
    'rfe': (RFE(DecisionTreeRegressor(random_state=RNG)), {'n_features_to_select': np.linspace(0, 1, 9, endpoint=True)[1:]})
}

# Defining models and their parameter space
MODEL_PARAMS = {
    'elastic_net': (ElasticNet(random_state=RNG,max_iter=10000, tol=1e-6), {'alpha': np.logspace(-4, 2, 5), 'l1_ratio': np.linspace(0, 1, 5)})
}


@ignore_warnings
def grid_search_n_best_model_eval(X_train, X_test, y_train, y_test, refit="R2", just_coefs=False, rng=RNG, only_iou=False):
    # Cross-validation procedure
    best_score = -np.inf
    best_method = None
    best_model = None
    best_params = None

    cv = KFold(n_splits=5, shuffle=True, random_state=rng)

    scoring = {"Explained Var.": make_scorer(explained_variance_score), "R2": make_scorer(r2_score), "MSE": make_scorer(mean_squared_error), "MAE": make_scorer(mean_absolute_error)}
    display_columns = ["%s_test_%s" % (s, m) for m in scoring.keys() for s in ["mean", "std"]] 

    results_all = dict()
    results = list()

    if not only_iou:
        for featsel_name, (featsel, fs_params) in FEATSEL_PARAMS.items():
            for model_name, (model, m_params) in MODEL_PARAMS.items():
                pipeline = Pipeline([
                    ('scaling', StandardScaler()),
                    ('featsel', featsel),
                    ('model', model)]
                    )
                
                param_grid = dict()
                for param_name, param_vals in fs_params.items():
                    param_grid[f"featsel__{param_name}"] = param_vals
                for param_name, param_vals in m_params.items():
                    param_grid[f"model__{param_name}"] = param_vals

                grid_search = GridSearchCV(pipeline, param_grid=param_grid, scoring=scoring, cv=cv, n_jobs=14, refit=refit).fit(X_train, y_train)

                results_all[f"{featsel_name}__{model_name}"] = grid_search.cv_results_

                display(pd.DataFrame(grid_search.cv_results_)[display_columns].sort_values("mean_test_%s" % refit, ascending=False).head(5).style.background_gradient(cmap="Blues").set_caption('Cross validation results'))

                results.append({
                    'pipeline_name': f"{featsel_name}__{model_name}",
                    refit: grid_search.best_score_,
                    'params': grid_search.best_params_
                })

                if grid_search.best_score_ > best_score:
                    best_score = grid_search.best_score_
                    best_method = featsel_name
                    best_model = model_name
                    best_params = grid_search.best_params_

        print("Best setup:")
        print(best_method, best_model, best_params, best_score)
        display(pd.DataFrame(results).sort_values(refit, ascending=False).style.background_gradient(cmap="Blues").set_caption('Cross validation results'))

        best_pipeline = Pipeline([
            ('scaling', StandardScaler()), 
            ('featsel', FEATSEL_PARAMS[best_method][0]),  
            ('model', MODEL_PARAMS[best_model][0])
            ]).set_params(**best_params)

        best_pipeline.fit(X_train, y_train)
        
        # Extracting the names of the selected features
        selected_features = X_train.columns[best_pipeline['featsel'].get_support()]
    else:
        for model_name, (model, m_params) in MODEL_PARAMS.items():
            pipeline = Pipeline([
                 ('scaling', StandardScaler()), 
                 ('model', model)
                ])
            
            param_grid = dict()
            for param_name, param_vals in m_params.items():
                param_grid[f"model__{param_name}"] = param_vals

            grid_search = GridSearchCV(pipeline, param_grid=param_grid, scoring=scoring, cv=cv, n_jobs=14, refit=refit).fit(X_train, y_train)

            results_all[f"{model_name}"] = grid_search.cv_results_

            display(pd.DataFrame(grid_search.cv_results_)[display_columns].sort_values("mean_test_%s" % refit, ascending=False).head(5).style.background_gradient(cmap="Blues").set_caption('Cross validation results'))

            results.append({
                'pipeline_name': f"{model_name}",
                refit: grid_search.best_score_,
                'params': grid_search.best_params_
            })

            if grid_search.best_score_ > best_score:
                best_score = grid_search.best_score_
                best_model = model_name
                best_params = grid_search.best_params_

        print("Best setup:")
        print(best_model, best_params, best_score)
        display(pd.DataFrame(results).sort_values(refit, ascending=False).style.background_gradient(cmap="Blues").set_caption('Cross validation results'))

        best_pipeline = Pipeline([
            ('scaling', StandardScaler()), 
            ('model', MODEL_PARAMS[best_model][0])
            ]).set_params(**best_params)
        
        best_pipeline.fit(X_train, y_train)
        
        # Extracting the names of the selected features
        selected_features = X_train.columns

    # Evaluate on held out test set
    y_pred = best_pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    ev = explained_variance_score(y_test, y_pred)

    model_perf = pd.DataFrame([{
        "Test MSE": mse,
        "Test MAE": mae,
        "Test Explained Var.": ev,
        "Test R2": r2,
        "CV %s" % refit: best_score
    }])

    display(model_perf)

    # Extracting the coefficients from the ElasticNet model
    coefficients = best_pipeline.named_steps['model'].coef_
    
    # Creating a DataFrame for visualization
    feature_importances = pd.DataFrame({'Feature': selected_features, 'Importance': coefficients})
    non_zero_importances = feature_importances[feature_importances['Importance'] != 0]

    if just_coefs:
        return feature_importances, model_perf

    # Plotting the feature importances
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=non_zero_importances.sort_values(by='Importance', ascending=False))
    plt.title('Feature Importances from ElasticNet Model')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.show()


@ignore_warnings
def repeated_grid_search_n_best_model_eval(X_train, y_train, X_test, y_test, refit="R2", 
                                           iou_feature=None, only_iou=False, no_feat_sel=False, **kwargs):
    if only_iou:
        X_train = X_train[[iou_feature]]
        X_test = X_test[[iou_feature]]

    feature_importances = []
    model_performances = []
    for rs in range(10):
        fe, mp = grid_search_n_best_model_eval(
            X_train, X_test, y_train.to_numpy()[:, 0], y_test.to_numpy()[:, 0], refit, True, rng=rs, only_iou=only_iou or no_feat_sel
            )
        feature_importances.append(fe)
        model_performances.append(mp)
    return pd.concat(feature_importances, axis=0), pd.concat(model_performances, axis=0)
