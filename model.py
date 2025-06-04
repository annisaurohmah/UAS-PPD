import pandas as pd
import numpy as np
from scipy import stats
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv('data/real_estate.csv', delimiter=',')
df = df.map(lambda x: float(str(x).replace(",", ".")) if isinstance(x, str) and x.replace(",", ".").replace(".", "", 1).isdigit() else x)



# Fungsi untuk penanganan outlier dengan IQR
def handle_outliers_iqr(X, columns):
    X = X.copy()
    for col in columns:
        Q1 = X[col].quantile(0.25)
        Q3 = X[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        X[col] = np.where(X[col] < lower_bound, lower_bound, X[col])
        X[col] = np.where(X[col] > upper_bound, upper_bound, X[col])
    return X

# Fungsi untuk penanganan outlier dengan Z-Score
def handle_outliers_zscore(X, columns, threshold=3):
    X = X.copy()
    for col in columns:
        z_scores = np.abs(stats.zscore(X[col]))
        mean = X[col].mean()
        std = X[col].std()
        lower_bound = mean - threshold * std
        upper_bound = mean + threshold * std
        X[col] = np.where(z_scores > threshold,
                          np.where(X[col] > mean, upper_bound, lower_bound),
                          X[col])
    return X

# Fungsi untuk log transformasi
def log_transform(X, columns):
    X = X.copy()
    for col in columns:
        X[col] = np.log1p(X[col])  # log(1 + x) untuk menghindari log(0)
    return X

# Pipeline untuk preprocessing (tanpa model)
preprocessing_pipeline = Pipeline([
    ('outlier_iqr', FunctionTransformer(lambda X: handle_outliers_iqr(X, columns=['X3 distance to the nearest MRT station']), validate=False)),
    ('outlier_zscore', FunctionTransformer(lambda X: handle_outliers_zscore(X, columns=['X5 latitude', 'X6 longitude']), validate=False)),
    ('log_transform', FunctionTransformer(lambda X: log_transform(X, columns=['X2 house age', 'X3 distance to the nearest MRT station']), validate=False)),
    ('scaler', StandardScaler())
])

# Pipeline untuk preprocessing target (hanya outlier handling untuk Y)
target_preprocessing_pipeline = Pipeline([
    ('outlier_iqr', FunctionTransformer(lambda X: handle_outliers_iqr(X, columns=['Y house price of unit area']), validate=False))
])

# Data preparation
df_X = df.drop(['No', 'Y house price of unit area'], axis=1)
df_y = df['Y house price of unit area']

# Periksa kolom kategorikal
cats = df_X.select_dtypes(include=['object', 'bool']).columns

# Pastikan tidak ada kolom kategorikal yang tidak terdeteksi (jika ada, perlu encoding)
if len(cats) > 0:
    raise ValueError("Terdapat kolom kategorikal yang perlu diencode sebelum preprocessing.")

# Terapkan preprocessing pipeline ke df_X
df_X = preprocessing_pipeline.fit_transform(df_X)
# Ubah kembali ke DataFrame agar sel-sel berikutnya tidak perlu diubah
df_X = pd.DataFrame(df_X, columns=df.drop(['No', 'Y house price of unit area'], axis=1).columns)

# Terapkan preprocessing ke df_y (hanya outlier handling)
df_y = target_preprocessing_pipeline.fit_transform(pd.DataFrame(df_y, columns=['Y house price of unit area']))['Y house price of unit area']

X = df_X.astype(float).values
y = df_y.astype(float).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor()
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n=== Evaluation Metrics: RandomForestRegressi  80% training 20% testing ===")
print(f"MAE  (Mean Absolute Error):      {mae:.2f}")
print(f"MSE  (Mean Squared Error):      {mse:.2f}")
print(f"RMSE (Root Mean Squared Error): {rmse:.2f}")
print(f"R²   (R-squared):               {r2:.2f}")