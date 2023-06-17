import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression, Lasso, Ridge
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2, RFE

FLAG_SHOW_PLOTS = True

# Helper methods
def validate_model(y_test: pd.Series, predictions: np.ndarray) -> None:

    # Calculate indicators
    cm = confusion_matrix(y_test, predictions)
    TN, FP, FN, TP = confusion_matrix(y_test, predictions).ravel()
    ACC = (TP + TN)/(TN + FP + FN + TP)

    print('True Positive(TP)  = ', TP)
    print('False Positive(FP) = ', FP)
    print('True Negative(TN)  = ', TN)
    print('False Negative(FN) = ', FN)
    print(f"ACC = {ACC}")

    # Compute RMSE
    rmse = mean_squared_error(y_test, predictions, squared=False)
    print(f"RMSE = {rmse}")

# Features selection functions
def select_correlation_matrix(X_train: pd.DataFrame, \
                              X_test: pd.DataFrame, \
                              y_train: pd.DataFrame, N: int) \
                                -> tuple[pd.DataFrame, pd.DataFrame]:
    # Display features correlation heatmap
    corr_matrix=pd.concat([X_train, y_train], axis=1).corr()

    if FLAG_SHOW_PLOTS:
        plt.figure(figsize=(25,25))
        sns.heatmap(corr_matrix,cmap='coolwarm')
        plt.show()

    target_matrix = corr_matrix.loc[:,'MFS_cens']

    # Select N best features
    top_N_val=target_matrix.nlargest(N+1)[1:]

    top_N_val_labels = top_N_val.index.tolist()
    print(top_N_val_labels)

    X_train_sel = X_train.loc[:, top_N_val_labels]
    X_test_sel = X_test.loc[:, top_N_val_labels]

    return X_train_sel, X_test_sel

def select_KBest(X_train: pd.DataFrame, \
                 X_test: pd.DataFrame, \
                 y_train: pd.DataFrame, \
                 N: int) \
                    -> tuple[pd.DataFrame, pd.DataFrame]:
    
    best_features = SelectKBest(chi2, k=N)
    best_features.fit(X_train, y_train)

    mask=best_features.get_support()

    X_train_sel=X_train.loc[:,mask]
    X_test_sel=X_test.loc[:,mask]

    features_to_process=X_train.columns[mask]
    # print(features_to_process)
    return X_train_sel, X_test_sel

def select_Ridge(X_train: pd.DataFrame, \
                 X_test:  pd.DataFrame, \
                 y_train: pd.DataFrame, \
                 N: int) \
                 -> tuple[pd.DataFrame, pd.DataFrame]:
    
    regressor = Ridge(alpha=1)
    regressor.fit(X_train, y_train)
    important_features = np.where(regressor.coef_ > 0)[0]
    important_features = np.sort(important_features)[:N]
    print(f"important features {important_features}")
    column_names = X_train.columns

    # Plot the features
    if FLAG_SHOW_PLOTS:
        fig = plt.figure(figsize=(10, 10))
        plt.bar(column_names,regressor.coef_)
        plt.xticks(rotation=90)
        plt.show()

    X_train_sel = X_train.iloc[:, important_features]
    X_test_sel = X_test.iloc[:, important_features]

    return X_train_sel, X_test_sel

def select_RFE(X_train: pd.DataFrame, \
                 X_test: pd.DataFrame, \
                 y_train: pd.DataFrame, \
                 N: int) \
                    -> tuple[pd.DataFrame, pd.DataFrame]:
    
    rfe = RFE(LogisticRegression(), n_features_to_select=N)
    X_train_sel = rfe.fit_transform(X_train, y_train)

    # Get the selected feature indices
    selected_feature_indices = rfe.get_support(indices=True)

    X_train_sel = X_train.iloc[:, selected_feature_indices]
    X_test_sel = X_test.iloc[:, selected_feature_indices]

    return X_train_sel, X_test_sel

# Classification methods
def classificator_LogisticRegression(X_train: pd.DataFrame, \
                                     X_test:  pd.DataFrame, \
                                     y_train: pd.Series, \
                                     y_test:  pd.Series) -> None:
    print(type(y_train))
    # Logistic regression
    model = LogisticRegression(random_state=0).fit(X_train, y_train)

    # Predykcja dla zbioru testowego
    y_pred = model.predict(X_test)

    # Evaluate the model
    validate_model(y_test, y_pred)

def classificator_Lasso(X_train: pd.DataFrame, \
                        X_test:  pd.DataFrame, \
                        y_train: pd.Series, \
                        y_test:  pd.Series) -> None:

    regressor = Lasso(alpha=0.1)
    y_pred = regressor.fit(X_train, y_train).predict(X_test)

    y_pred_binary = [1 if pred >= 0.5 else 0 for pred in y_pred]
    
    validate_model(y_test, y_pred_binary)
    
def classificator_RandomForest(X_train: pd.DataFrame, \
                               X_test:  pd.DataFrame, \
                               y_train: pd.Series, \
                               y_test:  pd.Series) -> None:
    
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}

    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestClassifier()
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 10, cv = 3, verbose=2, random_state=42, n_jobs = -1)
    
    # Fit the random search model
    y_pred = rf_random.fit(X_train, y_train).predict(X_test)

    validate_model(y_test, y_pred)

# Load .tsv
df = pd.read_table("clinical_radiomics.tsv")

# mfs - MFS, mfs_cens 
df_mfs = df.loc[:,["MFS", "MFS_cens"]]

# feature normalization
to_normalize=df.loc[:,'original_shape_Elongation':'original_ngtdm_Strength']
scalers={}

for col in to_normalize.columns:
    scaler=MinMaxScaler()
    scaler.fit(df[[col]])
    scalers[col]=scaler

for col in to_normalize.columns:
    scaler=scalers[col]
    df[col]=scaler.transform(df[[col]])

# create data set and target set
target=df.loc[:,'MFS_cens']

data = pd.concat([df.loc[:,'original_shape_Elongation':]], axis=1)

# Split the data into test and training set
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=10)

# Number of best features to select
N=5

# Select best features (pick one to test)
# X_train_sel, X_test_sel = select_correlation_matrix(X_train, X_test, y_train, N)
# X_train_sel, X_test_sel = select_KBest(X_train, X_test, y_train, N)
X_train_sel, X_test_sel = select_Ridge(X_train, X_test, y_train, N)
# X_train_sel, X_test_sel = select_RFE(X_train, X_test, y_train, N)

# Classificate
print("\n----- ----- Logistic Regression ----- ----- ")
classificator_LogisticRegression(X_train_sel, X_test_sel, y_train, y_test)
print("\n----- ----- LASSO ----- ----- ")
classificator_Lasso(X_train_sel, X_test_sel, y_train, y_test)
print("\n----- ----- Random Forest ----- ----- ")
classificator_RandomForest(X_train_sel, X_test_sel, y_train, y_test)
print("\n----- ----- ----- ----- ----- ")
