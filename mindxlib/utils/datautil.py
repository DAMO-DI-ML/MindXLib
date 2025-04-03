from pathlib import Path
import json
from typing import Dict, Tuple, Union, Optional
import warnings
import numpy as np
import pandas as pd

class DatasetLoader(object):
    def __init__(self, name: str, basedir: str='datasets'):
        path = Path(basedir) / name
        with open(path / 'meta.json') as f:
            meta = json.load(f)
        df = pd.read_csv(path / 'data.csv')
        label = df.eval(meta['positive']).to_numpy().astype(int)
        df.drop(meta['label'], axis='columns', inplace=True)
        df['label'] = label
    
        self._name = name
        self._meta = meta
        self._df = df

    @property
    def name(self) -> str:
        return self._name
    
    @property
    def meta(self) -> Dict:
        return self._meta
    
    @property
    def dataframe(self) -> pd.DataFrame:
        return self._df

def process_input_data(X: Union[pd.DataFrame, np.ndarray], 
                      y: Optional[Union[pd.DataFrame, pd.Series, np.ndarray]] = None,
                      feature_prefix: str = 'f',
                      feature_binarizer = None,
                      is_fit: bool = False) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], list, Optional[str]]:
    """Process input data into standardized format.
    
    Args:
        X: Input features (DataFrame or ndarray)
        y: Target labels (DataFrame, Series, ndarray, or None)
        feature_prefix: Prefix for feature names when using numpy arrays
        feature_binarizer: Optional feature binarizer instance
        is_fit: Whether to fit the feature binarizer (if provided)
        
    Returns:
        tuple: (X, y, feature_columns, label_column)
            - X: Processed feature DataFrame
            - y: Processed label DataFrame or None
            - feature_columns: List of feature column names
            - label_column: Name of label column or None
            
    Raises:
        ValueError: If input format is invalid
    """
    # Convert numpy arrays to pandas
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[f'{feature_prefix}{i}' for i in range(X.shape[1])])
    if isinstance(y, np.ndarray):
        if y is not None:
            if len(y.shape) > 1 and y.shape[1] != 1 and y.shape[0] != 1:
                raise ValueError(f'Ambiguous label column! Label array has shape {y.shape}')
            y = pd.Series(y.reshape(-1), name='label')

    if not isinstance(X, pd.DataFrame):
        raise ValueError('X must be DataFrame or ndarray')

    # Binarize features if enabled
    if feature_binarizer is not None:
        if is_fit:
            feature_binarizer.fit(X)
        X = feature_binarizer.transform(X)
        X.columns = [''.join(col).strip() for col in X.columns.values]

    feature_columns = list(X.columns)

    if y is None:
        label_column = None
        y_processed = None
    elif isinstance(y, pd.DataFrame):
        label_column = list(y.columns)
        if len(label_column) > 1:
            warnings.warn(f'Multiple label columns found ({len(label_column)} columns). Using first column.')
        label_column = label_column[0]
        y_processed = y
    elif isinstance(y, pd.Series):
        label_column = y.name
        y_processed = y.to_frame()
    else:
        raise ValueError('y must be DataFrame, Series or ndarray')

    return X, y_processed, feature_columns, label_column

def validate_shap_values(shap_values, class_index):
    """Validate SHAP values shape and class index, returning the appropriate values.
    
    Args:
        shap_values: The SHAP values array
        class_index: Index of the class to explain
        
    Returns:
        tuple: (values, is_multiclass)
            - values: The appropriate SHAP values to use
            - is_multiclass: Boolean indicating if values have multiple classes
            
    Raises:
        ValueError: If class_index is out of range
    """
    if len(shap_values.shape) == 3:
        if class_index >= shap_values.shape[2]:
            raise ValueError(f"class_index is out of range, got {class_index}, expected range is 0 to {shap_values.shape[2]-1}")
        try:
            return shap_values[:,:,class_index], True
        except:
            print(f"Error: shap_values has 3 dimensions, but class_index is out of range, got {class_index}, expected range is 0 to {shap_values.shape[2]-1}")
            return shap_values, True
    else:
        print(f"class_index is {class_index} but not used, because the prediction has only 2 dimensions")
        return shap_values, False
