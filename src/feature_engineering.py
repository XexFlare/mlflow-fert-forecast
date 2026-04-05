import pandas as pd


def encode_categorical(df):
    """
    Convert categorical variables into numeric format
    """
    print("Encoding categorical variables")

    return pd.get_dummies(df, drop_first=True)


def select_features(df, target_column="churn"):
    """
    Split features and target
    """
    print("Selecting features and target")

    if target_column not in df.columns:
        raise ValueError(f"{target_column} not found in dataset")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    return X, y