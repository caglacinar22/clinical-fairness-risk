import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_raw_data(path="../data/raw/heart_failure.csv"):
    df = pd.read_csv(path)
    return df


def preprocess(df):
    df['age_group'] = pd.cut(df['age'],
                              bins=[0, 50, 60, 70, 100],
                              labels=['<50', '50-60', '60-70', '70+'])

    X = df.drop(columns=['DEATH_EVENT', 'age_group'])
    y = df['DEATH_EVENT']
    sensitive = df[['sex', 'age_group']]

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    return X_scaled, y, sensitive


def split_data(X, y, sensitive, test_size=0.2, random_state=42):
    return train_test_split(
        X, y, sensitive,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )