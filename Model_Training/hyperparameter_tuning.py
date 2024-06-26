# hyperparameter_tuning.py

from sklearn.model_selection import GridSearchCV

def perform_hyperparameter_tuning(model, param_grid, X_train, y_train):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    return best_model
