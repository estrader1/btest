import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from joblib import Parallel, delayed


def calculate_simple_returns(df):
    """Helper function to calculate returns and excess returns"""
    df = df.copy()
    df['stock_return'] = df.groupby('ID')['close'].pct_change()
    df['idx_return'] = df.groupby('ID')['idx_close'].pct_change()
    df['stock_excess_return'] = df['stock_return'] - df['idx_return']

    return df

def rolling_residual_variance(df, window_size, dependent_var, independent_vars):
    """
    Performs rolling regression in parallel using joblib.

    Args:
        df: Pandas DataFrame containing the data.
        window_size: Size of the rolling window.
        dependent_var: Name of the dependent variable column.
        independent_vars: List of names of independent variable columns.

    Returns:
        Pandas DataFrame with the regression coefficients for each window.
        Returns None if there are issues.
    """

    n_rows = len(df)
    results = []

    def _regress_window(i):
        if i < window_size -1:
            return None # Handle edge cases at beginning of dataframe
        window_data = df.iloc[i - window_size + 1:i + 1]

        X = window_data[independent_vars].values
        y = window_data[dependent_var].values.reshape(-1,1) # Reshape y for sklearn

        if len(window_data) < window_size or np.any(np.isnan(X)) or np.any(np.isnan(y)):
            return None  # Handle cases where the window is incomplete or contains NaNs.
        
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)

        # Step 3: Calculate the residuals
        residuals = y - y_pred

        # Step 4: Compute the residual variance
        residual_variance = np.var(residuals, ddof=1) 

        return {'index': df.index[i], 'residual_variance': residual_variance} # Include index for proper merging


    results = Parallel(n_jobs=-1)(delayed(_regress_window)(i) for i in range(n_rows))
    
    

    # Filter out None results (from edge cases or NaN windows)
    valid_results = [r for r in results if r is not None]

    if not valid_results: # Check if all results are invalid
        return None

    results_df = pd.DataFrame(valid_results)#.set_index('index')
    
    # Handle multiindex
    if isinstance(results_df['index'].iloc[0], tuple):
        results_df[list(df.index.names)] = results_df['index'].apply(pd.Series)
        results_df = results_df.drop(columns='index').set_index(list(df.index.names))
    else:
        results_df = results_df.set_index('index')
        
    return results_df

def rolling_regression(df, window_size, dependent_var, independent_vars, reg_type='OLS', alpha=1.0):
    """
    Performs rolling regression in parallel using joblib, with options for OLS, Ridge, and Lasso.
    Returns np.nan for coefficients when X or y contains NaN values.

    Args:
        df: Pandas DataFrame containing the data.
        window_size: Size of the rolling window.
        dependent_var: Name of the dependent variable column.
        independent_vars: List of names of independent variable columns.
        reg_type: Type of regression to perform ('OLS', 'Ridge', 'Lasso'). Default is 'OLS'.
        alpha: Regularization strength for Ridge and Lasso. Default is 1.0.

    Returns:
        Pandas DataFrame with the regression coefficients for each window, indexed by the original DataFrame's index.
        Returns None if there are issues.
    """

    n_rows = len(df)
    results = []

    def _regress_window(i):
        if i < window_size - 1:
            return {'index': df.index[i],
                   'intercept': np.nan,
                   **dict(zip(independent_vars, [np.nan] * len(independent_vars)))}

        window_data = df.iloc[i - window_size + 1:i + 1]

        X = window_data[independent_vars].values
        y = window_data[dependent_var].values.reshape(-1,1)

        # Return NaN coefficients if window contains NaN or is incomplete
        if len(window_data) < window_size or np.any(np.isnan(X)) or np.any(np.isnan(y)):
            return {'index': df.index[i],
                   'intercept': np.nan,
                   **dict(zip(independent_vars, [np.nan] * len(independent_vars)))}

        if reg_type.upper() == 'OLS':
            model = LinearRegression()
        elif reg_type.upper() == 'RIDGE':
            model = Ridge(alpha=alpha)
        elif reg_type.upper() == 'LASSO':
            model = Lasso(alpha=alpha)
        else:
            raise ValueError("Invalid reg_type. Choose 'OLS', 'Ridge', or 'Lasso'.")

        model.fit(X, y)
        coefs = model.coef_.flatten()
        intercept = model.intercept_

        return {'index': df.index[i], 'intercept': intercept, **dict(zip(independent_vars, coefs))}

    results = Parallel(n_jobs=-1)(delayed(_regress_window)(i) for i in range(n_rows))

    # All results should be valid now since we're returning NaN instead of None
    results_df = pd.DataFrame(results)

    # Handle multiindex
    if isinstance(results_df['index'].iloc[0], tuple):
        results_df[list(df.index.names)] = results_df['index'].apply(pd.Series)
        results_df = results_df.drop(columns='index').set_index(list(df.index.names))
    else:
        results_df = results_df.set_index('index')

    return results_df

