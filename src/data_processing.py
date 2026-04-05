import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split


def load_data(data_path):
    """
    Load FarmersWorld customer data
    """
    print(f"Loading data from {data_path}")

    file_path = os.path.join(data_path, "farmersworld_fertilizer_customers.csv")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")

    df = pd.read_csv(file_path)

    print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
    return df


def clean_data(df):
    """
    Clean and preprocess the data
    """
    print("Cleaning data")

    df_clean = df.copy()

    # Handle missing values
    num_cols = df_clean.select_dtypes(include=['number']).columns
    for col in num_cols:
        if df_clean[col].isna().sum() > 0:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    cat_cols = df_clean.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if df_clean[col].isna().sum() > 0:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

    # Safe filtering (only if columns exist)
    if 'farm_size_hectares' in df_clean.columns:
        df_clean = df_clean[df_clean['farm_size_hectares'] <= 50]

    if 'months_since_first_purchase' in df_clean.columns:
        df_clean = df_clean[df_clean['months_since_first_purchase'] >= 1]

    print(f"After cleaning: {df_clean.shape[0]} rows remaining")
    return df_clean


def preprocess_data(df):
    """
    Feature engineering
    """
    print("Preprocessing data")

    df = df.copy()

    # Convert categoricals safely
    for col in ['district', 'region', 'payment_method', 'crop_type']:
        if col in df.columns:
            df[col] = df[col].astype('category')

    # Derived features (safe checks)
    if all(col in df.columns for col in ['avg_fertilizer_kg', 'farm_size_hectares']):
        df['fertilizer_per_hectare'] = df['avg_fertilizer_kg'] / df['farm_size_hectares']

    if all(col in df.columns for col in ['total_purchases', 'months_since_first_purchase']):
        df['purchase_frequency'] = df['total_purchases'] / df['months_since_first_purchase']

    if all(col in df.columns for col in ['support_tickets', 'months_since_first_purchase']):
        df['support_interaction_rate'] = df['support_tickets'] / df['months_since_first_purchase']

    # Age groups
    if 'age' in df.columns:
        df['age_group'] = pd.cut(
            df['age'],
            bins=[18, 30, 45, 60, 100],
            labels=['Young', 'Adult', 'Middle-aged', 'Senior']
        )

    # Farm size categories
    if 'farm_size_hectares' in df.columns:
        df['farm_size_category'] = pd.cut(
            df['farm_size_hectares'],
            bins=[0, 1, 5, 10, 50],
            labels=['Very Small', 'Small', 'Medium', 'Large']
        )

    return df


def split_data(df, target_column="churn"):
    """
    Split dataset into train/test
    """
    print("Splitting data")

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    return train_test_split(X, y, test_size=0.2, random_state=42)