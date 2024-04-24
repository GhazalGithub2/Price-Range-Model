import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
#Read the data and see the headers
data= pd.read_csv('train - train.csv')
data.head(5)
# Summarize central tendency like(count,mean,std....)
data.describe()
#Print information:
data.info()
#Data cleaning
#--Drop unnecessary columns
#-- Remove duplicates row
# data=data.drop_duplicates()
# data.info()
#--Fill null values
# Define columns based on their types
numeric_columns = ['battery_power', 'clock_speed', 'fc', 'int_memory', 'm_dep', 'mobile_wt', 'pc', 'px_height', 'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time']
binary_columns = ['blue', 'dual_sim', 'four_g', 'three_g', 'touch_screen', 'wifi']
ordinal_column = ['n_cores']

# Impute missing values
for col in numeric_columns:
    data[col].fillna(data[col].mean(), inplace=True)

for col in binary_columns:
    data[col].fillna(data[col].mode()[0], inplace=True)

for col in ordinal_column:
    data[col].fillna(data[col].mode()[0], inplace=True)

# Confirm that there are no more missing values
print(data.isnull().sum())
# Data Transformation


# Separate features and target variable
X = data.drop(columns=[ 'price_range'])  # Exclude target variable 'price_range'
# Convert n_cores column to integer in the training dataset to match the test one
X['n_cores'] = X['n_cores'].astype(int)
X['four_g'] = X['four_g'].astype(int)
y = data['price_range']
numeric_columns = [col for col in X.columns if col not in binary_columns + ordinal_column]

# Define transformers
binary_transformer = OneHotEncoder(drop='first')  # Drop first category to avoid multicollinearity
ordinal_transformer = OneHotEncoder(drop='first')
numeric_transformer = StandardScaler()
# Define column transformer for encoding categorical variables and scaling numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ('numeric', numeric_transformer, numeric_columns),  # Scale numerical features first
        ('binary', binary_transformer, binary_columns),
        ('ordinal', ordinal_transformer, ordinal_column)
    ])

# Fit and transform data
X_transformed = preprocessor.fit_transform(X)
# Define column names after transformation
transformed_columns = (list(preprocessor.named_transformers_['numeric'].get_feature_names_out(numeric_columns)) +
                       list(preprocessor.named_transformers_['binary'].get_feature_names_out(binary_columns)) +
                       list(preprocessor.named_transformers_['ordinal'].get_feature_names_out(ordinal_column)))

# Convert transformed data back to DataFrame for further analysis
X_transformed_df = pd.DataFrame(X_transformed, columns=transformed_columns)

# Check the shape and first few rows of the transformed DataFrame
print("Shape of transformed data:", X_transformed_df.shape)
print(X_transformed_df.head())

# Divide train data


# Split the preprocessed data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_transformed_df, y, test_size=0.2, random_state=42)
# Model Construction


# Define the models
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier()
}

# Train and evaluate each model
for name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy:.4f}")

#More Enhancement


# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter tuning
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}  # Adjust C values as needed
grid_search = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)
best_model = grid_search.best_estimator_

# Model evaluation
y_pred = best_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("Logistic Regression Accuracy:", accuracy)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()