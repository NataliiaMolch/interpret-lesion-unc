import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score, f1_score, accuracy_score, make_scorer


# Global random state range
RNG_rf = np.random.RandomState(0)

def grid_search_n_best_model_eval_rf(X_train, X_test, y_train, y_test, refit="R2", just_coefs=False, rng=RNG_rf):

    # Define the pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor(random_state=42))
    ])

    # Define the extended parameter grid
    param_grid = {
        'rf__n_estimators': [20, 50, 100, 200, 300],
        'rf__max_features': ['auto', 'sqrt', 'log2', 0.2, 0.5, 0.8],
        'rf__max_depth': [None, 10, 20, 30, 40, 50],
        'rf__min_samples_split': [2, 5, 10],
        'rf__min_samples_leaf': [1, 2, 4],
        'rf__bootstrap': [True, False]
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=rng)


    scoring = {"Explained Var.": make_scorer(explained_variance_score), "R2": make_scorer(r2_score), "MSE": make_scorer(mean_squared_error), "MAE": make_scorer(mean_absolute_error)}

    # Set up GridSearchCV
    grid_search = GridSearchCV(pipeline, param_grid=param_grid, scoring=scoring, cv=cv, n_jobs=14, refit=refit).fit(X_train, y_train)

    # Get the best model
    best_model = grid_search.best_estimator_

    # Extract feature importances
    feature_importances = best_model.named_steps['rf'].feature_importances_

    # Create a DataFrame for feature importances
    features_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': feature_importances
    })

    # Evaluate on held out test set
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    ev = explained_variance_score(y_test, y_pred)

    model_perf = pd.DataFrame([{
        "Test MSE": mse,
        "Test MAE": mae,
        "Test Explained Var.": ev,
        "Test R2": r2,
        "CV %s" % refit: grid_search.best_score_
    }])

    # Sort by importance
    features_df = features_df.sort_values(by='Importance', ascending=False)

    if just_coefs:
        return features_df, model_perf

    # Plot feature importances
    plt.figure(figsize=(10, 6))
    plt.bar(features_df['Feature'], features_df['Importance'])
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.title('Feature Importances from Random Forest')
    plt.xticks(rotation=45)
    plt.show()
