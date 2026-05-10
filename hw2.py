import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import zscore

def load_data (file_path):
    df = pd.read_csv(file_path)
    return df

def clear_data (df):
    df_clean = df.copy()

    for col in ['Alley', 'Pool QC', 'Fence']:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna("None")

    for col in ['Bsmt Full Bath', 'Garage Area']:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(0)

    df_clean['Lot Frontage'] = df_clean.groupby('Neighborhood')['Lot Frontage'].transform(
        lambda x: x.fillna(x.median())
    )

    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    df_clean = pd.get_dummies(df_clean, columns=categorical_cols, drop_first=True)

    return df_clean

def ridge_analysis(df):
    X = df.drop(columns=['SalePrice'])
    y = df['SalePrice']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)

    coef_abs = pd.Series(abs(ridge.coef_), index=X.columns)
    top10 = coef_abs.sort_values(ascending=False).head(10)
    print(top10)

def anomaly_analysis(df_original, df_clean):

    plt.figure(figsize=(10, 6))
    plt.scatter(df_original['Gr Liv Area'], df_original['SalePrice'])
    plt.xlabel('Gr Liv Area')
    plt.ylabel('SalePrice')
    plt.title('Цена vs Жилая площадь')
    plt.show()

    z_scores = np.abs(
        zscore(df_original[['Gr Liv Area', 'SalePrice']])
    )

    filtered_entries = (z_scores < 3).all(axis=1)

    df_no_outliers = df_clean[filtered_entries]

    print('\nРазмер данных ДО удаления:', df_clean.shape)
    print('Размер данных ПОСЛЕ удаления:', df_no_outliers.shape)



    X = df_clean.drop(columns=['SalePrice'])
    y = df_clean['SalePrice']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model_before = LinearRegression()

    model_before.fit(X_train, y_train)

    pred_before = model_before.predict(X_test)

    rmse_before = np.sqrt(
        mean_squared_error(y_test, pred_before)
    )

    r2_before = r2_score(y_test, pred_before)

    X2 = df_no_outliers.drop(columns=['SalePrice'])
    y2 = df_no_outliers['SalePrice']

    X2_train, X2_test, y2_train, y2_test = train_test_split(
        X2,
        y2,
        test_size=0.2,
        random_state=42
    )

    model_after = LinearRegression()

    model_after.fit(X2_train, y2_train)

    pred_after = model_after.predict(X2_test)

    rmse_after = np.sqrt(
        mean_squared_error(y2_test, pred_after)
    )

    r2_after = r2_score(y2_test, pred_after)

    print('\n===== ДО удаления аномалий =====')
    print('RMSE:', rmse_before)
    print('R2:', r2_before)

    print('\n===== ПОСЛЕ удаления аномалий =====')
    print('RMSE:', rmse_after)
    print('R2:', r2_after)

def clustering(df):

    X = df.drop(columns=['SalePrice'])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=5, random_state=42)

    clusters = kmeans.fit_predict(X_scaled)

    df['Cluster'] = clusters

    print('\nКоличество объектов в сегментах:')
    print(df['Cluster'].value_counts())

    plt.figure(figsize=(10, 6))
    plt.scatter(df['Gr Liv Area'], df['Overall Qual'], c=df['Cluster'])
    plt.xlabel('Gr Liv Area')
    plt.ylabel('Overall Qual')
    plt.title('Кластеры домов')
    plt.show()

def pca_regression(df):

    X = df.drop(columns=['SalePrice'])
    y = df['SalePrice']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=0.95)

    X_pca = pca.fit_transform(X_scaled)

    print('\nКоличество компонент PCA:', pca.n_components_)

    X_train, X_test, y_train, y_test = train_test_split(
        X_pca,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    print('\n===== PCA Regression =====')
    print('RMSE:', np.sqrt(mean_squared_error(y_test, pred)))
    print('R2:', r2_score(y_test, pred))

def time_analysis(df_original):

    df = df_original.copy()

    # Возраст дома
    df['HouseAgeAtSale'] = df['Yr Sold'] - df['Year Built']

    # Лет после ремонта
    df['YearsSinceRemodel'] = df['Yr Sold'] - df['Year Remod/Add']

    # Средняя цена по годам
    yearly_prices = df.groupby('Yr Sold')['SalePrice'].mean()

    plt.figure(figsize=(8, 5))
    yearly_prices.plot(marker='o')
    plt.title('Средняя цена по годам')
    plt.ylabel('SalePrice')
    plt.grid()
    plt.show()

    # Средняя цена по месяцам
    monthly_prices = df.groupby('Mo Sold')['SalePrice'].mean()

    plt.figure(figsize=(10, 5))
    monthly_prices.plot(kind='bar')
    plt.title('Средняя цена по месяцам')
    plt.ylabel('SalePrice')
    plt.show()

    print('\nСредняя цена по годам:')
    print(yearly_prices)

    print('\nСредняя цена по месяцам:')
    print(monthly_prices)

    # Кризис 2008
    before_2008 = df[df['Yr Sold'] < 2008]['SalePrice'].mean()
    after_2008 = df[df['Yr Sold'] >= 2008]['SalePrice'].mean()

    drop_percent = ((before_2008 - after_2008) / before_2008) * 100

    print('\nСредняя цена ДО 2008:', before_2008)
    print('Средняя цена ПОСЛЕ 2008:', after_2008)
    print(f'Падение цен: {drop_percent:.2f}%')

def main():
    file_path = "AmesHousing.csv"
    df = load_data(file_path)
    df_clean = clear_data(df)

    ridge_analysis(df_clean)

    anomaly_analysis(df, df_clean)

    clustering(df_clean)

    pca_regression(df_clean)

    time_analysis(df)

if __name__ == "__main__":
    main()