# Import the required libraries
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from ML_Pipeline.utils import read_data, inspection, null_values
from ML_Pipeline.encoding import encode_categories
from ML_Pipeline.stats_model import logistic_regression
from ML_Pipeline.ml_model import prepare_model, run_model
from ML_Pipeline.evaluate_metrics import confusion_matrix, roc_curve
from ML_Pipeline.imbalanced_data import run_model_bweights, run_model_aweights, adjust_imbalance, prepare_model_smote
from ML_Pipeline.feature_engg import var_threshold_selection, rfe_selection

# Step 1: Read the initial dataset
df = read_data("../input/data_regression.csv")

# Step 2: Data Inspection and Cleaning
x = inspection(df)

# Step 3: Handling Missing Values
df = null_values(df)

# Step 4: Encoding Categorical Variables
categorical_vars = ['gender', 'multi_screen', 'mail_subscribed']
df = encode_categories(df, categorical_vars)

# Step 5: Run the Logistic Regression model with statsmodel
model_stats = logistic_regression(df, class_col='churn', cols_to_exclude=['customer_id', 'phone_no', 'year'])
pickle.dump(model_stats, open('../output/model_stats.pkl', 'wb'))

# Step 6: Run the Logistic Regression model with sklearn

# Selecting only the numerical columns and excluding specific columns
X_train, X_test, y_train, y_test = prepare_model(df, class_col='churn',
                                               cols_to_exclude=['customer_id', 'phone_no', 'year'])

# Run the model
model_log, y_pred = run_model(X_train, X_test, y_train, y_test)

# Step 7: Performance Metrics
conf_matrix = confusion_matrix(y_test, y_pred)
# print(conf_matrix)
roc_val = roc_curve(model_log, X_test, y_test)

# Save the model
pickle.dump(model_log, open('../output/model1.pkl', 'wb'))

# Step 8: Dealing with Imbalanced Data

# Model with default parameters for balancing the data
balanced_model1 = run_model_bweights(X_train, X_test, y_train, y_test)
pickle.dump(balanced_model1, open('../output/balanced1_model.pkl', 'wb'))

# Model with user-defined weights for balancing the data
balanced_model2 = run_model_aweights(X_train, X_test, y_train, y_test, {0: 90, 1: 10})
pickle.dump(balanced_model2, open('../output/balanced2_model.pkl', 'wb'))

# Model with adjusting the imbalanced data
resampled_df = adjust_imbalance(X_train, y_train, class_col='churn')
X_train, X_test, y_train, y_test = prepare_model(resampled_df, class_col='churn',
                                                 cols_to_exclude=['customer_id', 'phone_no', 'year'])
adj_bal = run_model(X_train, X_test, y_train, y_test)
pickle.dump(adj_bal, open('../output/adjusted_model.pkl', 'wb'))

# Model with SMOTE
X_train, X_test, y_train, y_test = prepare_model_smote(df, class_col='churn',
                                                      cols_to_exclude=['customer_id', 'phone_no', 'year'])
smote_model, y_pred = run_model(X_train, X_test, y_train, y_test)
pickle.dump(smote_model, open('../output/smote_model.pkl', 'wb'))

# Step 9: Perform Feature Engineering

# Perform feature selection with variance threshold
var_feature = var_threshold_selection(df, cols_to_exclude=['customer_id', 'phone_no', 'year'],
                                      class_col='churn', threshold=1)

# Create a Logistic Regression model with selected variance features
X_train, X_test, y_train, y_test = prepare_model(resampled_df, class_col='churn',
                                                cols_to exclude=['customer_id', 'phone_no', 'year',
                                                               'gender', 'age', 'no_of_days_subscribed',
                                                               'multi_screen', 'mail_subscribed',
                                                               'minimum_daily_mins', 'weekly_max_night_mins',
                                                               'videos_watched', 'customer_support_calls',
                                                               'churn', 'gender_code', 'multi_screen_code',
                                                               'mail_subscribed_code'])
model_var_feat = run_model(X_train, X_test, y_train, y_test)
pickle.dump(model_var_feat, open('../output/model_var_feat.pkl', 'wb'))

# Perform feature selection with Recursive Feature Elimination (RFE)
rfe_sel = rfe_selection(df, class_col='churn',
                        cols_to_exclude=['customer_id', 'phone_no', 'year'],
                        model=model_log)
X_train, X_test, y_train, y_test = prepare_model(resampled_df, class_col='churn',
                                                cols_to_exclude=['customer_id', 'phone_no', 'year',
                                                               'gender', 'age', 'no_of_days_subscribed',
                                                               'multi_screen', 'mail_subscribed',
                                                               'weekly_max_night_mins',
                                                               'gender_code', 'multi_screen_code',
                                                               'mail_subscribed_code'])
model_rfe_feat = run_model(X_train, X_test, y_train, y_test)
pickle.dump(model_rfe_feat, open('../output/model_rfe_feat.pkl', 'wb'))
