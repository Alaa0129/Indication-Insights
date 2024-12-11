# %%
# Sampling libraries
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# SalesInsightsPreprocessing
from PreprocessingScript import *

# Set print options
pd.set_option('display.max_columns', None)

df = mergedHospitalPharmacySickHouseDfYearAndSortedWithTypes


# %% [markdown]
# #### Define features/independent variables 'X', and specify our target/dependent variable, y

# %%
X = df.values
y = df['Volume'].values

# %%
# split data into test and train - 80/20
X_split_train, X_split_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert numpy arrays to DataFrames
X_split_train_df = pd.DataFrame(X_split_train, columns=df.columns)
X_split_test_df = pd.DataFrame(X_split_test, columns=df.columns)

X_train = X_split_train_df.drop(['Volume', 'Account Description', 'Size', 'Year Month (after 2000) in Datetime', 'Value'], axis=1).values
X_test = X_split_test_df.drop(['Volume', 'Account Description', 'Size', 'Year Month (after 2000) in Datetime', 'Value'], axis=1).values

# %% [markdown]
# #### Add more random values to df

# %%
from sklearn.utils import resample
import pandas as pd

# Convert y_train to Series if it's not already
if not isinstance(y_train, pd.Series):
    y_train = pd.Series(y_train)

def create_resampled_dataframe(X_train, y_train, n_samples, df):
    # Resample the data to get additional samples
    X_resampled, y_resampled = resample(X_train, y_train, replace=True, n_samples=n_samples, random_state=42)

    # Convert the resampled arrays to DataFrames and Series
    X_resampled_df = pd.DataFrame(X_resampled, columns=df.columns)
    y_resampled_df = pd.Series(y_resampled)

    # Concatenate the original data with the additional samples
    X_train_extended = pd.concat([pd.DataFrame(X_train, columns=df.columns), X_resampled_df], axis=0)
    y_train_extended = pd.concat([y_train, y_resampled_df], axis=0)

    # Create a DataFrame with the extended data
    df_extended = pd.DataFrame(X_train_extended, columns=df.columns)
    return df_extended

# Create DataFrames with 20000, 50000, and 100000 additional rows
df_resampled_20000 = create_resampled_dataframe(X_split_train, y_train, 20000, df)
df_resampled_50000 = create_resampled_dataframe(X_split_train, y_train, 50000, df)
df_resampled_100000 = create_resampled_dataframe(X_split_train, y_train, 100000, df)

# %% [markdown]
# #### Show all dataframes after sampling etc

# %%
# L04AC05 = 0
# Hospital = 0
# Pharmacy = 1

df_C05 = df[df['WHO ATC 5 Code'] == 0]
df_20000_C05 = df_resampled_20000[df_resampled_20000['WHO ATC 5 Code'] == 0]
df_50000_C05 = df_resampled_50000[df_resampled_50000['WHO ATC 5 Code'] == 0]
df_100000_C05 = df_resampled_100000[df_resampled_100000['WHO ATC 5 Code'] == 0]

df_c05_hospitals = df_C05[df_C05['Type'] == 0]
df_c05_pharmacies = df_C05[df_C05['Type'] == 1]

df_20000_c05_hospitals = df_20000_C05[df_20000_C05['Type'] == 0]
df_20000_c05_pharmacies = df_20000_C05[df_20000_C05['Type'] == 1]

df_50000_c05_hospitals = df_50000_C05[df_50000_C05['Type'] == 0]
df_50000_c05_pharmacies = df_50000_C05[df_50000_C05['Type'] == 1]

df_100000_res_c05_hospitals = df_100000_C05[df_100000_C05['Type'] == 0]
df_100000_res_c05_pharmacies = df_100000_C05[df_100000_C05['Type'] == 1]

print('df:')
print('Total rows of L04AC05:', len(df_C05))
print('Total rows of L04AC05 in hospitals:', len(df_c05_hospitals))
print('Total rows of L04AC05 in pharmacies:', len(df_c05_pharmacies))
print()
print('resampled 20k df:')
print('Total rows of L04AC05:', len(df_20000_C05))
print('Total rows of L04AC05 in hospitals:', len(df_20000_c05_hospitals))
print('Total rows of L04AC05 in pharmacies:', len(df_20000_c05_pharmacies))
print()
print('resampled 50k df:')
print('Total rows of L04AC05:', len(df_50000_C05))
print('Total rows of L04AC05 in hospitals:', len(df_50000_c05_hospitals))
print('Total rows of L04AC05 in pharmacies:', len(df_50000_c05_pharmacies))
print()
print('resampled 100k df:')
print('Total rows of L04AC05:', len(df_100000_C05))
print('Total rows of L04AC05 in hospitals:', len(df_100000_res_c05_hospitals))
print('Total rows of L04AC05 in pharmacies:', len(df_100000_res_c05_pharmacies))

# %%
# L04AB02 = 1
# Hospital = 0
# Pharmacy = 1

df_B02 = df[df['WHO ATC 5 Code'] == 1]
df_20000_B02 = df_resampled_20000[df_resampled_20000['WHO ATC 5 Code'] == 1]
df_50000_B02 = df_resampled_50000[df_resampled_50000['WHO ATC 5 Code'] == 1]
df_100000_B02 = df_resampled_100000[df_resampled_100000['WHO ATC 5 Code'] == 1]

df_b02_hospitals = df_B02[df_B02['Type'] == 0]
df_b02_pharmacies = df_B02[df_B02['Type'] == 1]

df_20000_b02_hospitals = df_20000_B02[df_20000_B02['Type'] == 0]
df_20000_b02_pharmacies = df_20000_B02[df_20000_B02['Type'] == 1]

df_50000_b02_hospitals = df_50000_B02[df_50000_B02['Type'] == 0]
df_50000_b02_pharmacies = df_50000_B02[df_50000_B02['Type'] == 1]

df_100000_b02_hospitals = df_100000_B02[df_100000_B02['Type'] == 0]
df_100000_b02_pharmacies = df_100000_B02[df_100000_B02['Type'] == 1]

print('df:')
print('Total rows of L04AB02:', len(df_B02))
print('Total rows of L04AB02 in hospitals:', len(df_b02_hospitals))
print('Total rows of L04AB02 in pharmacies:', len(df_b02_pharmacies))
print()
print('resampled 20k df:')
print('Total rows of L04AB02:', len(df_20000_B02))
print('Total rows of L04AB02 in hospitals:', len(df_20000_b02_hospitals))
print('Total rows of L04AB02 in pharmacies:', len(df_20000_b02_pharmacies))
print()
print('resampled 50k df:')
print('Total rows of L04AB02:', len(df_50000_B02))
print('Total rows of L04AB02 in hospitals:', len(df_50000_b02_hospitals))
print('Total rows of L04AB02 in pharmacies:', len(df_50000_b02_pharmacies))
print()
print('resampled 100k df:')
print('Total rows of L04AB02:', len(df_100000_B02))
print('Total rows of L04AB02 in hospitals:', len(df_100000_b02_hospitals))
print('Total rows of L04AB02 in pharmacies:', len(df_100000_b02_pharmacies))

# %%
# L04AB05 = 2
# Hospital = 0
# Pharmacy = 1

df_B05 = df[df['WHO ATC 5 Code'] == 2]
df_20000_B05 = df_resampled_20000[df_resampled_20000['WHO ATC 5 Code'] == 2]
df_50000_B05 = df_resampled_50000[df_resampled_50000['WHO ATC 5 Code'] == 2]
df_100000_B05 = df_resampled_100000[df_resampled_100000['WHO ATC 5 Code'] == 2]

df_b05_hospitals = df_B05[df_B05['Type'] == 0]
df_b05_pharmacies = df_B05[df_B05['Type'] == 1]

df_20000_b05_hospitals = df_20000_B05[df_20000_B05['Type'] == 0]
df_20000_b05_pharmacies = df_20000_B05[df_20000_B05['Type'] == 1]

df_50000_b05_hospitals = df_50000_B05[df_50000_B05['Type'] == 0]
df_50000_b05_pharmacies = df_50000_B05[df_50000_B05['Type'] == 1]

df_100000_b05_hospitals = df_100000_B05[df_100000_B05['Type'] == 0]
df_100000_b05_pharmacies = df_100000_B05[df_100000_B05['Type'] == 1]

print('df:')
print('Total rows of L04AB05:', len(df_B05))
print('Total rows of L04AB05 in hospitals:', len(df_b05_hospitals))
print('Total rows of L04AB05 in pharmacies:', len(df_b05_pharmacies))
print()
print('resampled 20k df:')
print('Total rows of L04AB05:', len(df_20000_B05))
print('Total rows of L04AB05 in hospitals:', len(df_20000_b05_hospitals))
print('Total rows of L04AB05 in pharmacies:', len(df_20000_b05_pharmacies))
print()
print('resampled 50k df:')
print('Total rows of L04AB05:', len(df_50000_B05))
print('Total rows of L04AB05 in hospitals:', len(df_50000_b05_hospitals))
print('Total rows of L04AB05 in pharmacies:', len(df_50000_b05_pharmacies))
print()
print('resampled 100k df:')
print('Total rows of L04AB05:', len(df_100000_B05))
print('Total rows of L04AB05 in hospitals:', len(df_100000_b05_hospitals))
print('Total rows of L04AB05 in pharmacies:', len(df_100000_b05_pharmacies))

# %% [markdown]
# #### Summarization of the dataframes we will use

# %%
X_train
y_train

X_res_20k = df_resampled_20000.drop(['Volume', 'Account Description', 'Size', 'Year Month (after 2000) in Datetime', 'Value'], axis=1).values
y_res_20k = df_resampled_20000['Volume'].values

X__res_50k = df_resampled_50000.drop(['Volume', 'Account Description', 'Size', 'Year Month (after 2000) in Datetime', 'Value'], axis=1).values
y_res_50k = df_resampled_50000['Volume'].values

X_res_100k = df_resampled_100000.drop(['Volume', 'Account Description', 'Size', 'Year Month (after 2000) in Datetime', 'Value'], axis=1).values
y_res_100k = df_resampled_100000['Volume'].values


