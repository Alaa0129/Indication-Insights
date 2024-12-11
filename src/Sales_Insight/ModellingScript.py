# %%
# SalesInsightsPreprocessing
from SamplingScript import *

# libraries to visualize
import matplotlib.pyplot as plt

# Import regression models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.metrics import accuracy_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

## some stuff for making pretty plots
from matplotlib import rcParams
from cycler import cycler

### Tensorflow and Keras imports ###
# Import TensorFlow for machine learning
import tensorflow as tf
# Import Keras for high-level neural networks API
from tensorflow import keras
# Import Dense and Activation layers for neural network architecture
from tensorflow.keras.layers import Dense, Activation
# Import Sequential for linear stacking of layers
from keras.models import Sequential
# Import KerasClassifier to make Keras models compatible with scikit-learn
from scikeras.wrappers import KerasClassifier, KerasRegressor

from matplotlib import cm
# Visualization tool for the elbow method to determine the optimal number of clusters
from yellowbrick.cluster.elbow import kelbow_visualizer

# Calculates the Silhouette Score which measures the quality of clusters
from sklearn.metrics import silhouette_score
# KMeans clustering algorithm
from sklearn.cluster import KMeans, DBSCAN
# Splits data into random train and test subsets
from sklearn.model_selection import train_test_split

## Set plotting style and print options
sns.set_theme()
sns.set_style("whitegrid")
sns.set_context("paper") #Possible are paper, notebook, talk and poster
# Set figure format
wide_format, square_format = False, True
if wide_format:
    d = {'lines.linewidth': 2, 'axes.titlesize': 18, 'axes.labelsize': 18, 'xtick.labelsize': 12, 'ytick.labelsize': 12,\
     'legend.fontsize': 15, 'font.family': 'serif', 'figure.figsize': (9,6)}
if square_format:
    d = {'lines.linewidth': 2, 'axes.titlesize': 16, 'axes.labelsize': 14, 'xtick.labelsize': 10, 'ytick.labelsize': 10,\
     'legend.fontsize': 12, 'font.family': 'serif', 'figure.figsize': (6,6)}
    
d_colors = {'axes.prop_cycle': cycler(color = ['teal', 'navy', 'coral', 'plum', 'purple', 'olivedrab',\
         'black', 'red', 'cyan', 'yellow', 'khaki','lightblue'])}
rcParams.update(d)
rcParams.update(d_colors)

# %% [markdown]
# #### Extra functions

# %%
training_data = [(X_train, y_train), (X_res_20k, y_res_20k), (X__res_50k, y_res_50k), (X_res_100k, y_res_100k), (X_res_500k, y_res_500k)]
titles = ['OG df', '20k df', '50k df', '100k df', '500k df']
dataframes = [df, df_resampled_20000, df_resampled_50000, df_resampled_100000, df_resampled_500000]

# %%
def calculate_relative_errors(y_prediction):
    non_zero_mask = (y_test != 0)
    rel_error = np.mean(np.abs(y_test[non_zero_mask] - y_prediction[non_zero_mask]) / y_test[non_zero_mask])
    rel_error_med = np.median(np.abs(y_test[non_zero_mask] - y_prediction[non_zero_mask]) / y_test[non_zero_mask])
    rel_quan = np.quantile(np.abs(y_test[non_zero_mask] - y_prediction[non_zero_mask]) / y_test[non_zero_mask], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    
    return rel_error, rel_error_med, rel_quan

# %%
def plot_true_vs_predicted(y_test, yPredLst, rSqrdLst):
    fig, ax = plt.subplots(ncols=len(titles), figsize=(15,5))
    ax = ax.flatten() 
    fig.suptitle('True vs predicted')

    for i in range(5):
        ax[i].plot(y_test, yPredLst[i], '.', markersize=1.5, alpha=.6)
        ax[i].set(xlabel='True value', ylabel='Predicted value', title=titles[i], ylim=[0,5000], xlim=[0,5000])
        ax[i].text(1000, 4000, rf'$R^2$ = {rSqrdLst[i]:.3f}', fontsize=12)

    fig.tight_layout()
    plt.show()

# %%
def plot_feature_importance(featImportLst):
    fig, ax = plt.subplots(ncols=len(titles), figsize=(15, 5))
    ax = ax.flatten()
    fig.suptitle('Feature Importance for Different Dataframes')

    for i in range(5):
        features = dataframes[i].drop(['Volume', 'Account Description', 'Size', 'Year Month (after 2000) in Datetime', 'Value'], axis=1).columns
        importances = featImportLst[i]
        indices = np.argsort(importances)
        
        ax[i].barh(features[indices], importances[indices])
        ax[i].set(xlabel='Importance', ylabel='Feature', title=titles[i])

    fig.tight_layout()
    plt.show()

# %%
def plot_evaluation_results(history):

    # make a list of the train and val metrics
    metrics = list(history.history.keys())
    
    # make lists of train and val metrics
    val_metrics = [entry for entry in metrics if entry.startswith('val_')]
    train_metrics = [entry for entry in metrics if not entry.startswith('val_')]

    # the number of metrics to plot
    Nmetrics = len(val_metrics)
    width = 6 * Nmetrics

    fig, ax = plt.subplots(ncols = Nmetrics, figsize=(width, 6))

    # plot the train and val results for each metric
    for i, axx in enumerate(ax):
        ax[i].plot(history.history[train_metrics[i]], label='train', alpha = 0.7)  
        ax[i].plot(history.history[val_metrics[i]], label='test', alpha = 0.7)
        ax[i].set_ylabel(f'{train_metrics[i].capitalize()}')
        ax[i].set_xlabel('Epoch')
        ax[i].legend(loc='best')
    fig.tight_layout()
    return fig, ax 

# %%
def evaluate_regression_results(model, X_train, y_train,\
                                           metrics = [r2_score, mean_absolute_error], metric_names = ['r2', 'MAE']):
    
    # make predictions
    y_pred_train = model.predict(X_train, verbose = 0)   
    y_pred_val = model.predict(X_test, verbose = 0)

    # calculate metrics
    for metric, metric_name in zip(metrics, metric_names):
        metric(y_train, y_pred_train.round())
        print(f'{metric_name} on training data: {metric(y_train, y_pred_train.round()):.3f}')
        print(f'{metric_name} on validation data: {metric(y_test, y_pred_val.round()):.3f}')
        return

# %% [markdown]
# #### Linear Regression

# %%
lin_reg = LinearRegression()

r_sqrd_list_linreg = []
y_pred_list_linreg = []
perm_import_list_linreg = []

for X, y in training_data:
    # fit and predict
    lin_reg.fit(X, y)
    y_pred = lin_reg.predict(X_test)
    
    # calculate the R^2 score
    rsquared = lin_reg.score(X_test, y_test)
    rel_error, rel_error_med, rel_quan = calculate_relative_errors(y_pred)
    perm_importance = permutation_importance(lin_reg, X_test, y_test, scoring='neg_mean_squared_error')

    y_pred_list_linreg.append(y_pred)
    r_sqrd_list_linreg.append(rsquared)
    perm_import_list_linreg.append(perm_importance.importances_mean)
    print(f'Relative error: {rel_error * 100 :.3f} %')
    print(f'Relative median error: {rel_error_med * 100 :.3f} %')
    print(rel_quan * 100, '%')
    print(f'R squared: {rsquared:.3f}')
    print('')

# %%
plot_true_vs_predicted(y_test, y_pred_list_linreg, r_sqrd_list_linreg)

# %%
plot_feature_importance(perm_import_list_linreg)

# %% [markdown]
# #### Random Forest with hyperparameter tuning

# %%
ran_for_reg = RandomForestRegressor(max_depth=df.columns.size, random_state=42)

y_pred_list_ranfor = []
r_sqrd_list_ranfor = []
feat_import_list_ranfor = []

for X, y in training_data:
    # fit and predict
    ran_for_reg.fit(X, y)
    y_pred = ran_for_reg.predict(X_test)

    # calculate the R^2 score
    rsquared = ran_for_reg.score(X_test, y_test)

    non_zero_mask = (y_test != 0)
    rel_error, rel_error_med, rel_quan = calculate_relative_errors(y_pred)
    
    y_pred_list_ranfor.append(y_pred)
    r_sqrd_list_ranfor.append(rsquared)
    feat_import_list_ranfor.append(ran_for_reg.feature_importances_)
    print(f'Relative error: {rel_error * 100 :.3f} %')
    print(f'Relative median error: {rel_error_med * 100 :.3f} %')
    print(f'Relative quantile error: {rel_quan * 100} %')
    print(f'R squared: {rsquared:.3f}')
    print()

# %%
plot_true_vs_predicted(y_test, y_pred_list_ranfor, r_sqrd_list_ranfor)

# %%
plot_feature_importance(feat_import_list_ranfor)

# %% [markdown]
# #### XGBoost

# %%
xgb = XGBRegressor()

# %%
y_pred_list_xgb = []
r_sqrd_list_xgb = []
feat_import_list_xgb = []

for x,y in training_data:
    xgb.fit(x, y)
    y_pred_xgb = xgb.predict(X_test)
    rsquared_xgb = xgb.score(X_test, y_test)
    non_zero_mask = (y_test != 0)
    rel_error, rel_error_med, rel_quan = calculate_relative_errors(y_pred_xgb)

    y_pred_list_xgb.append(y_pred_xgb)
    r_sqrd_list_xgb.append(rsquared_xgb)
    feat_import_list_xgb.append(xgb.feature_importances_)
    print(f'Relative error: {rel_error * 100 :.3f} %')
    print(f'Relative median error: {rel_error_med * 100 :.3f} %')
    print(rel_quan * 100, '%')
    print(f'R squared: {rsquared_xgb:.3f}')
    print('')

# %%
plot_true_vs_predicted(y_test, y_pred_list_xgb, r_sqrd_list_xgb)

# %%
plot_feature_importance(feat_import_list_xgb)

# %% [markdown]
# #### TODO: 
# Lav boxplot der viser resultatet for hver model
# 


