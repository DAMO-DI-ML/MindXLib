import pandas as pd
import os

def tic_tac_toe():
    """Load and return the tic-tac-toe dataset.
    
    The tic-tac-toe dataset contains 958 instances of possible tic-tac-toe board configurations
    at the end of the game, where 'x' is assumed to have played first.
    
    Returns
    -------
    X : pandas.DataFrame
        The feature matrix with 9 columns representing the board positions
    y : pandas.Series
        The target vector where 'positive' means 'x' won and 'negative' means 'o' won
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, 'tic_tac_toe.csv')
    
    data = pd.read_csv(data_path, header=None)
    y = data.iloc[:,-1]
    X = data.iloc[:,:-1]
    
    return X, y 