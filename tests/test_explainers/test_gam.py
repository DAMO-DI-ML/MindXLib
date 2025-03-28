import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mindxlib.explainers.interactive_gam.gam import GAM
from sklearn.metrics import mean_squared_error, r2_score
from mindxlib.visualization.interactive import create_app

def generate_synthetic_data(n_samples=1000, noise_level=0.1, random_state=42):
    """
    y = x1 + 10*x2^2 - x3^3 + e
    """
    np.random.seed(random_state)
    
    # Generate features
    x1 = np.random.uniform(-1, 10, n_samples)
    x2 = np.random.uniform(-10, 1, n_samples)
    x3 = np.random.normal(-15, 15, n_samples)
    
    # Generate noise
    noise = np.random.normal(0, noise_level, n_samples)
    
    # Generate target
    y = x1 + 10 * (x2 ** 2) - np.sin(x3 ** 3) + noise
    
    # Create DataFrame
    X = pd.DataFrame({
        'x1': x1,
        'x2': x2,
        'x3': x3
    })
    
    return X, y

def test_pandas_basic(X,y):
    """Test basic functionality of the GAM explainer"""
    # Generate synthetic data
    
    
    # Split into train and test sets
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Create and fit GAM model
    gam = GAM(max_iter=200, lambda_1=0.01, verbose=True, bin_num=64)
    gam.fit(X_train, y_train)
    
    # gam.show(X_test,mode='interactive')
    y_pred = gam.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # The model should achieve reasonable performance
    assert r2 > 0.8, f"R² score too low: {r2}"
    

    
    # # Check that the shape functions match our expectations
    # shape_functions = gam.get_shape_functions()
    
    # # x1 should be approximately linear
    # x1_values, x1_contributions = shape_functions['x1']
    # x1_corr = np.corrcoef(x1_values, x1_contributions)[0, 1]
    # assert abs(x1_corr) > 0.9, f"x1 shape function not linear enough: {x1_corr}"
    
    # # x2 should be approximately quadratic
    # # We can check this by fitting a quadratic function and comparing
    # x2_values, x2_contributions = shape_functions['x2']
    # sorted_indices = np.argsort(x2_values)
    # x2_sorted = x2_values[sorted_indices]
    # y2_sorted = x2_contributions[sorted_indices]
    
    # # Fit quadratic function
    # coeffs = np.polyfit(x2_sorted, y2_sorted, 2)
    # y2_fit = np.polyval(coeffs, x2_sorted)
    # x2_r2 = 1 - np.sum((y2_sorted - y2_fit)**2) / np.sum((y2_sorted - np.mean(y2_sorted))**2)
    # assert x2_r2 > 0.8, f"x2 shape function not quadratic enough: {x2_r2}"
    
    # # x3 should be approximately cubic
    # x3_values, x3_contributions = shape_functions['x3']
    # sorted_indices = np.argsort(x3_values)
    # x3_sorted = x3_values[sorted_indices]
    # y3_sorted = x3_contributions[sorted_indices]
    
    # # Fit cubic function
    # coeffs = np.polyfit(x3_sorted, y3_sorted, 3)
    # y3_fit = np.polyval(coeffs, x3_sorted)
    # x3_r2 = 1 - np.sum((y3_sorted - y3_fit)**2) / np.sum((y3_sorted - np.mean(y3_sorted))**2)
    # assert x3_r2 > 0.8, f"x3 shape function not cubic enough: {x3_r2}"
    
    print("All tests passed!")
    return gam

def test_numpy_basic():
    """Test basic functionality of the GAM explainer"""
    # Generate synthetic data
    X, y = generate_synthetic_data(n_samples=1000, noise_level=0.1)
    X = X.to_numpy()
    
    # Split into train and test sets
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Create and fit GAM model
    gam = GAM(max_iter=200, lambda_1=0.01, verbose=True, feature_prefix='feature_')
    gam.fit(X_train, y_train)
    
    # Make predictions
    y_pred = gam.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # The model should achieve reasonable performance
    assert r2 > 0.8, f"R² score too low: {r2}"
    
    # Get feature importance
    importance = gam.analyze_feature_importance()
    print("Feature Importance:")
    print(importance)
    return gam
def test_gam_constraints():
    """Test GAM with shape constraints"""
    # Generate synthetic data
    X, y = generate_synthetic_data(n_samples=1000, noise_level=0.1)
    
    # Split into train and test sets
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Create and fit GAM model with constraints
    gam = GAM(max_iter=200, lambda_1=0.01, verbose=True)
    
    # Add constraints before fitting
    
    # Fit the model
    gam.fit(X_train, y_train)
    y_pred = gam.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error with constraints: {mse:.4f}")
    print(f"R² Score with constraints: {r2:.4f}")

    gam.show(figsize=(5,10), layout=(3,1), xlim=[(-1,1),(0,1),(-30,0)], title='Before constraints')
        # x1 should be increasing
    gam.add_constraint(-1, 1, 'd', 'x1')
    
    # x2 should be convex
    gam.add_constraint(0, 1, 'c', 'x2')
    
    # x3 should be decreasing in the negative range
    gam.add_constraint(-30, 0, 'd', 'x3')
    gam.update(X_train, y_train)
    gam.show(figsize=(5,10), layout=(3,1), xlim=[(-1,1),(0,1),(-30,0)], title='After constraints')
    


    shape_functions = gam.get_shape_functions()
    
    # Check x1 is increasing
    x1_values, x1_contributions = shape_functions['x1']
    sorted_indices = np.argsort(x1_values)
    x1_sorted = x1_values[sorted_indices]
    y1_sorted = x1_contributions[sorted_indices]
    assert np.all(np.diff(y1_sorted[(x1_sorted > -1) & (x1_sorted < 1)]) <= 1e-10), "x1 constraint (decreasing) not satisfied"
    
    # Check x2 is convex
    x2_values, x2_contributions = shape_functions['x2']
    sorted_indices = np.argsort(x2_values)
    x2_sorted = x2_values[sorted_indices]
    y2_sorted = x2_contributions[sorted_indices]
    # For convexity, the second derivative should be non-negative
    second_diff = np.diff(np.diff(y2_sorted))
    assert np.all(second_diff[(x2_sorted[2:] > 0) & (x2_sorted[2:] < 1)] <= 1e-1), "x2 constraint (concave) not satisfied"
    
    # Check x3 is decreasing in the negative range
    x3_values, x3_contributions = shape_functions['x3']
    neg_mask = x3_values < 0
    x3_neg = x3_values[neg_mask]
    y3_neg = x3_contributions[neg_mask]
    sorted_indices = np.argsort(x3_neg)
    x3_neg_sorted = x3_neg[sorted_indices]
    y3_neg_sorted = y3_neg[sorted_indices]
    assert np.all(np.diff(y3_neg_sorted[(x3_neg_sorted > -30) & (x3_neg_sorted < 0)]) <= 1e-10), "x3 constraint (decreasing in negative range) not satisfied"
    
    print("All constraint tests passed!")
    return gam




    # Run basic test
X, y = generate_synthetic_data(n_samples=1000, noise_level=0.1)
gam = test_pandas_basic(X,y)
# gam = test_gam_constraints()
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
gam.show(data = X_test,mode='interactive',intercept=True)
# gam.show(data = X_test, mode='static')


