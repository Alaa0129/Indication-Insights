# %%
# Sampling libraries
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


# SalesInsightsPreprocessing
from PreprocessingScript import *

# Set print options
pd.set_option('display.max_columns', None)

df = mergedHospitalPharmacyDfYearAndSortedWithTypes

# Separate the rows where Type is 0 and 1
df_type_0 = df[df['Type'] == 0]
df_type_1 = df[df['Type'] == 1]


# %% [markdown]
# #### Define features/independent variables 'X', and specify our target/dependent variable, y

# %%
# Below, we make a list of features/independent variables 'X', and specify our target/dependent variable, y
# The model will guess/predict the 'y' feature (our target) based on the list of features, 'X'
# Running the cell will not produce any output. This is because we are defining X and y, which we will be using in the next section to train our model

X = df.drop(['Volume', 'Account Description', 'Size', 'Year Month (after 2000) in Datetime', 'Value'], axis=1).values
y = df['Volume'].values

X_type_0 = df_type_0.drop(['Volume', 'Account Description', 'Size', 'Year Month (after 2000) in Datetime', 'Value'], axis=1).values
y_type_0 = df_type_0['Volume'].values


# %%
# split data into test and train - 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_type_0, X_test_type_0, y_train_type_0, y_test_type_0 = train_test_split(X_type_0, y_type_0, test_size=0.2, random_state=42)

# %% [markdown]
# #### Oversample hospital values

# %%
# oversampling for df_type_0
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_train_type_0, y_train_type_0)

# %%
# make the resampled data into a dataframe
hospital_df_resampled = pd.DataFrame(X_resampled, columns=df.drop(['Volume', 'Account Description', 'Size', 'Year Month (after 2000) in Datetime', 'Value'], axis=1).columns)

print(len(hospital_df_resampled))
print(len(df_type_1))

oversampled_df = pd.concat([hospital_df_resampled, df_type_1], ignore_index=True)

# %% [markdown]
# #### Add more random values to df and oversampled_df

# %%
# Define the number of rows to generate
n_samples = 20000

# Generate additional rows for df and oversampled_df
df_resampled = resample(df, replace=True, n_samples=n_samples, random_state=42)
oversampled_df_resampled = resample(oversampled_df, replace=True, n_samples=n_samples, random_state=42)

# Concatenate the resampled dataframes
df_resampled = pd.concat([df, df_resampled], ignore_index=True)
oversampled_df_resampled = pd.concat([oversampled_df, oversampled_df_resampled], ignore_index=True)


# %% [markdown]
# #### Show all dataframes after sampling etc

# %%
# L04AC05 = 0
# Hospital = 0
# Pharmacy = 1

df_C05 = df[df['WHO ATC 5 Code'] == 0]
df_over_C05 = oversampled_df[oversampled_df['WHO ATC 5 Code'] == 0]
df_res_C05 = df_resampled[df_resampled['WHO ATC 5 Code'] == 0]
df_over_res_C05 = oversampled_df_resampled[oversampled_df_resampled['WHO ATC 5 Code'] == 0]

df_c05_hospitals = df_C05[df_C05['Type'] == 0]
df_c05_pharmacies = df_C05[df_C05['Type'] == 1]

df_over_c05_hospitals = df_over_C05[df_over_C05['Type'] == 0]
df_over_c05_pharmacies = df_over_C05[df_over_C05['Type'] == 1]

df_res_c05_hospitals = df_res_C05[df_res_C05['Type'] == 0]
df_res_c05_pharmacies = df_res_C05[df_res_C05['Type'] == 1]

df_over_res_c05_hospitals = df_over_res_C05[df_over_res_C05['Type'] == 0]
df_over_res_c05_pharmacies = df_over_res_C05[df_over_res_C05['Type'] == 1]


print('df:')
print('Total rows of L04AC05:', len(df_C05))
print('Total rows of L04AC05 in hospitals:', len(df_c05_hospitals))
print('Total rows of L04AC05 in pharmacies:', len(df_c05_pharmacies))
print()
print('oversampled df:')
print('Total rows of L04AC05:', len(df_over_C05))
print('Total rows of L04AC05 in hospitals:', len(df_over_c05_hospitals))
print('Total rows of L04AC05 in pharmacies:', len(df_over_c05_pharmacies))
print()
print('resampled df:')
print('Total rows of L04AC05:', len(df_res_C05))
print('Total rows of L04AC05 in hospitals:', len(df_res_c05_hospitals))
print('Total rows of L04AC05 in pharmacies:', len(df_res_c05_pharmacies))
print()
print('oversampled resampled df:')
print('Total rows of L04AC05:', len(df_over_res_C05))
print('Total rows of L04AC05 in hospitals:', len(df_over_res_c05_hospitals))
print('Total rows of L04AC05 in pharmacies:', len(df_over_res_c05_pharmacies))


# %%
# L04AB02 = 1
# Hospital = 0
# Pharmacy = 1

df_B02 = df[df['WHO ATC 5 Code'] == 1]
df_over_B02 = oversampled_df[oversampled_df['WHO ATC 5 Code'] == 1]
df_res_B02 = df_resampled[df_resampled['WHO ATC 5 Code'] == 1]
df_over_res_B02 = oversampled_df_resampled[oversampled_df_resampled['WHO ATC 5 Code'] == 1]

df_b02_hospitals = df_B02[df_B02['Type'] == 0]
df_b02_pharmacies = df_B02[df_B02['Type'] == 1]

df_over_b02_hospitals = df_over_B02[df_over_B02['Type'] == 0]
df_over_b02_pharmacies = df_over_B02[df_over_B02['Type'] == 1]

df_res_b02_hospitals = df_res_B02[df_res_B02['Type'] == 0]
df_res_b02_pharmacies = df_res_B02[df_res_B02['Type'] == 1]

df_over_res_b02_hospitals = df_over_res_B02[df_over_res_B02['Type'] == 0]
df_over_res_b02_pharmacies = df_over_res_B02[df_over_res_B02['Type'] == 1]

print('df:')
print('Total rows of L04AB02:', len(df_B02))
print('Total rows of L04AB02 in hospitals:', len(df_b02_hospitals))
print('Total rows of L04AB02 in pharmacies:', len(df_b02_pharmacies))
print()
print('oversampled df:')
print('Total rows of L04AB02:', len(df_over_B02))
print('Total rows of L04AB02 in hospitals:', len(df_over_b02_hospitals))
print('Total rows of L04AB02 in pharmacies:', len(df_over_b02_pharmacies))
print()
print('resampled df:')
print('Total rows of L04AB02:', len(df_res_B02))
print('Total rows of L04AB02 in hospitals:', len(df_res_b02_hospitals))
print('Total rows of L04AB02 in pharmacies:', len(df_res_b02_pharmacies))
print()
print('oversampled resampled df:')
print('Total rows of L04AB02:', len(df_over_res_B02))
print('Total rows of L04AB02 in hospitals:', len(df_over_res_b02_hospitals))
print('Total rows of L04AB02 in pharmacies:', len(df_over_res_b02_pharmacies))

# %%
# L04AB05 = 2
# Hospital = 0
# Pharmacy = 1

df_B05 = df[df['WHO ATC 5 Code'] == 2]
df_over_B05 = oversampled_df[oversampled_df['WHO ATC 5 Code'] == 2]
df_res_B05 = df_resampled[df_resampled['WHO ATC 5 Code'] == 2]
df_over_res_B05 = oversampled_df_resampled[oversampled_df_resampled['WHO ATC 5 Code'] == 2]

df_b05_hospitals = df_B05[df_B05['Type'] == 0]
df_b05_pharmacies = df_B05[df_B05['Type'] == 1]

df_over_b05_hospitals = df_over_B05[df_over_B05['Type'] == 0]
df_over_b05_pharmacies = df_over_B05[df_over_B05['Type'] == 1]

df_res_b05_hospitals = df_res_B05[df_res_B05['Type'] == 0]
df_res_b05_pharmacies = df_res_B05[df_res_B05['Type'] == 1]

df_over_res_b05_hospitals = df_over_res_B05[df_over_res_B05['Type'] == 0]
df_over_res_b05_pharmacies = df_over_res_B05[df_over_res_B05['Type'] == 1]

print('df:')
print('Total rows of L04AB05:', len(df_B05))
print('Total rows of L04AB05 in hospitals:', len(df_b05_hospitals))
print('Total rows of L04AB05 in pharmacies:', len(df_b05_pharmacies))
print()
print('oversampled df:')
print('Total rows of L04AB05:', len(df_over_B05))
print('Total rows of L04AB05 in hospitals:', len(df_over_b05_hospitals))
print('Total rows of L04AB05 in pharmacies:', len(df_over_b05_pharmacies))
print()
print('resampled df:')
print('Total rows of L04AB05:', len(df_res_B05))
print('Total rows of L04AB05 in hospitals:', len(df_res_b05_hospitals))
print('Total rows of L04AB05 in pharmacies:', len(df_res_b05_pharmacies))
print()
print('oversampled resampled df:')
print('Total rows of L04AB05:', len(df_over_res_B05))
print('Total rows of L04AB05 in hospitals:', len(df_over_res_b05_hospitals))
print('Total rows of L04AB05 in pharmacies:', len(df_over_res_b05_pharmacies))

# %% [markdown]
# #### Summarization of the dataframes we will use

# %%
# L04AC05
df_C05
df_over_C05
df_res_C05
df_over_res_C05

# L04AB02
df_B02
df_over_B02
df_res_B02
df_over_res_B02

# L04AB05
df_B05
df_over_B05
df_res_B05
df_over_res_B05


