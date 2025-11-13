from sklearn.ensemble import RandomForestRegressor

def train_random_forest(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def predict_rf(model, X_new):
    return model.predict(X_new)
