# %% [markdown]
# ## Relevant libraries and methods

# %%
import os
import glob

# Data handling and manipulation library
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Set print options
pd.set_option('display.max_columns', None)

# %%
pharDf = pd.read_csv('data_processed\sales_insights_pharmacy_cleaned.csv', delimiter=',')
hospDf = pd.read_csv('data_processed\sales_insights_hospital_cleaned.csv', delimiter=',')

# %%
print(len(pharDf))
print(len(hospDf))
print(pharDf.columns)
print(hospDf.columns)

# %% [markdown]
# #### Merge datasets phar and hosp insights and remove unwanted columns + duplicates

# %%
# Merge the two datasets together
merged_df = pd.concat([pharDf, hospDf], ignore_index=True)

# remove the column 'Region_Færøerne'
merged_df = merged_df.drop(columns=['Region_Færøerne'])

# drop duplicates in pharmacy_df
merged_df_no_Dups = merged_df.drop_duplicates()

# convert the float to int in column 'Year Month (after 2000)'.
# merged_df_no_Dups['Year Month (after 2000)'] = merged_df_no_Dups['Year Month (after 2000)'].astype(int)

print(len(merged_df_no_Dups))

# %% [markdown]
# #### Divide the dataset into two subdatasets: Hospital and Pharmacy

# %%
hospitalDf = merged_df_no_Dups[~merged_df_no_Dups['Account Description'].str.contains('Apotek', case=False) & ~merged_df_no_Dups['Account Description'].str.contains('fyrkilden', case=False)]
pharmacyDf = merged_df_no_Dups[(merged_df_no_Dups['Account Description'].str.contains('Apotek', case=False) | merged_df_no_Dups['Account Description'].str.contains('fyrkilden', case=False)) & ~merged_df_no_Dups['Account Description'].str.contains('Sygehus', case=False)]
sickHouseDf = merged_df_no_Dups[merged_df_no_Dups['Account Description'].str.contains('Sygehus', case=False)]

print(len(hospitalDf))
print(len(pharmacyDf))
print(len(sickHouseDf))

# %% [markdown]
# #### Convert Year Month to datetime object and remove year column

# %%
# Convert the 'Year Month (after 2000)' column to a datetime
hospitalDf['Year Month (after 2000) in Datetime'] = pd.to_datetime(hospitalDf['Year Month (after 2000)'].astype(int).astype(str), format='%y%m')
pharmacyDf['Year Month (after 2000) in Datetime'] = pd.to_datetime(pharmacyDf['Year Month (after 2000)'].astype(int).astype(str), format='%y%m')
sickHouseDf['Year Month (after 2000) in Datetime'] = pd.to_datetime(sickHouseDf['Year Month (after 2000)'].astype(int).astype(str), format='%y%m')

# Remove time from the datetime object
hospitalDf['Year Month (after 2000) in Datetime'] = hospitalDf['Year Month (after 2000) in Datetime'].dt.to_period('M')
pharmacyDf['Year Month (after 2000) in Datetime'] = pharmacyDf['Year Month (after 2000) in Datetime'].dt.to_period('M')
sickHouseDf['Year Month (after 2000) in Datetime'] = sickHouseDf['Year Month (after 2000) in Datetime'].dt.to_period('M')

# Convert the date object to a string
hospitalDf['Year Month (after 2000) in Datetime'] = hospitalDf['Year Month (after 2000) in Datetime'].astype(str)
pharmacyDf['Year Month (after 2000) in Datetime'] = pharmacyDf['Year Month (after 2000) in Datetime'].astype(str)
sickHouseDf['Year Month (after 2000) in Datetime'] = sickHouseDf['Year Month (after 2000) in Datetime'].astype(str)

# Add the 'Year Month (after 2000) in Datetime' column to the 8th column
cols = hospitalDf.columns.tolist()
cols = cols[:8] + cols[-1:] + cols[8:-1]
hospitalDf = hospitalDf[cols]
pharmacyDf = pharmacyDf[cols]
sickHouseDf = sickHouseDf[cols]

# drop the 'Year (after 2000)' column
hospitalDf = hospitalDf.drop(columns=['Year (after 2000)'])
pharmacyDf = pharmacyDf.drop(columns=['Year (after 2000)'])
sickHouseDf = sickHouseDf.drop(columns=['Year (after 2000)'])

# drop the 'month' column
hospitalDf = hospitalDf.drop(columns=['Month'])
pharmacyDf = pharmacyDf.drop(columns=['Month'])
sickHouseDf = sickHouseDf.drop(columns=['Month'])

# %% [markdown]
# #### Add new Account Description ID Column

# %%
# Convert all values in Account Description in pharmacy to a numbers and store in a new column 'Account Description Number'. 
# The numbers are ascending from 100 to the number of unique values in 'Account Description'
pharmacyDf['Account Description ID'] = pd.factorize(pharmacyDf['Account Description'])[0] + 100
pharmacyDf = pharmacyDf[['Account Description ID'] + [col for col in pharmacyDf.columns if col != 'Account Description ID']]

# Convert all values in Account Description in hospital to a numbers and store in a new column 'Account Description Number'. 
# The numbers are ascending from 0 to the number of unique values in 'Account Description'
hospitalDf['Account Description ID'] = pd.factorize(hospitalDf['Account Description'])[0]
hospitalDf = hospitalDf[['Account Description ID'] + [col for col in hospitalDf.columns if col != 'Account Description ID']]

# Convert all values in Account Description in sickhouses to a numbers and store in a new column 'Account Description Number'.
# The numbers are ascending from 1000 to the number of unique values in 'Account Description'
sickHouseDf['Account Description ID'] = pd.factorize(sickHouseDf['Account Description'])[0] + 1000
sickHouseDf = sickHouseDf[['Account Description ID'] + [col for col in sickHouseDf.columns if col != 'Account Description ID']]

# merge the three dataframes
hospitalPharmacySickHouseDf = pd.concat([hospitalDf, pharmacyDf, sickHouseDf], ignore_index=True)

# move the new column as the first column
hospitalPharmacySickHouseDf = hospitalPharmacySickHouseDf[['Account Description ID'] + [col for col in hospitalPharmacySickHouseDf.columns if col != 'Account Description ID']]

hospitalPharmacySickHouseDf

# %% [markdown]
# #### Sort hospital and pharmacy datasets into values between 2010 and 2019 and only for Stelara

# %%
hospitalDfYearSorted = hospitalDf[(hospitalDf['Year Month (after 2000)'] >= 1001) & (hospitalDf['Year Month (after 2000)'] <= 1912)]
PharmacyDfYearSorted = pharmacyDf[(pharmacyDf['Year Month (after 2000)'] >= 1001) & (pharmacyDf['Year Month (after 2000)'] <= 1912)]
sickHouseDfYearSorted = sickHouseDf[(sickHouseDf['Year Month (after 2000)'] >= 1001) & (sickHouseDf['Year Month (after 2000)'] <= 1912)]
mergedHospitalPharmacySickHouseDfYearSorted = pd.concat([hospitalDfYearSorted, PharmacyDfYearSorted, sickHouseDfYearSorted], ignore_index=True)

#Stelara
hospitalDfYearAndStelaraSorted = hospitalDfYearSorted[hospitalDfYearSorted['Product_Stelara'] == True]
PharmacyDfYearAndStelaraSorted = PharmacyDfYearSorted[PharmacyDfYearSorted['Product_Stelara'] == True]
sickHouseDfYearAndStelaraSorted = sickHouseDfYearSorted[sickHouseDfYearSorted['Product_Stelara'] == True]
mergedHospitalPharmacySickHouseDfYearAndStelaraSorted = pd.concat([hospitalDfYearAndStelaraSorted, PharmacyDfYearAndStelaraSorted, sickHouseDfYearAndStelaraSorted], ignore_index=True)

print('Stelara sales:')
print('Hospital total sales btw 2010 and 2019 for Stelara: ', sum(hospitalDfYearAndStelaraSorted['Volume']))
print('Pharmacy total sales btw 2010 and 2019 for Stelara: ', sum(PharmacyDfYearAndStelaraSorted['Volume']))
print('Sickhouse total sales btw 2010 and 2019 for Stelara: ', sum(sickHouseDfYearAndStelaraSorted['Volume']))
print()

#Cimzia
hospitalDfYearAndCimziaSorted = hospitalDfYearSorted[hospitalDfYearSorted['Product_Cimzia'] == True]
PharmacyDfYearAndCimziaSorted = PharmacyDfYearSorted[PharmacyDfYearSorted['Product_Cimzia'] == True]
sickHouseDfYearAndCimziaSorted = sickHouseDfYearSorted[sickHouseDfYearSorted['Product_Cimzia'] == True]
mergedHospitalPharmacySickHouseDfYearAndCimziaSorted = pd.concat([hospitalDfYearAndCimziaSorted, PharmacyDfYearAndCimziaSorted, sickHouseDfYearAndCimziaSorted], ignore_index=True)

print('Cimzia sales:')
print('Hospital total sales btw 2010 and 2019 for Cimzia: ', sum(hospitalDfYearAndCimziaSorted['Volume']))
print('Pharmacy total sales btw 2010 and 2019 for Cimzia: ', sum(PharmacyDfYearAndCimziaSorted['Volume']))
print('Sickhouse total sales btw 2010 and 2019 for Cimzia: ', sum(sickHouseDfYearAndCimziaSorted['Volume']))
print()

#Inflectra
hospitalDfYearAndInflectraSorted = hospitalDfYearSorted[hospitalDfYearSorted['Product_Inflectra'] == True]
PharmacyDfYearAndInflectraSorted = PharmacyDfYearSorted[PharmacyDfYearSorted['Product_Inflectra'] == True]
sickHouseDfYearAndInflectraSorted = sickHouseDfYearSorted[sickHouseDfYearSorted['Product_Inflectra'] == True]
mergedHospitalPharmacySickHouseDfYearAndInflectraSorted = pd.concat([hospitalDfYearAndInflectraSorted, PharmacyDfYearAndInflectraSorted, sickHouseDfYearAndInflectraSorted], ignore_index=True)

print('Inflectra sales:')
print('Hospital total sales btw 2010 and 2019 for Inflectra: ', sum(hospitalDfYearAndInflectraSorted['Volume']))
print('Pharmacy total sales btw 2010 and 2019 for Inflectra: ', sum(PharmacyDfYearAndInflectraSorted['Volume']))
print('Sickhouse total sales btw 2010 and 2019 for Inflectra: ', sum(sickHouseDfYearAndInflectraSorted['Volume']))
print()

#Remicade
hospitalDfYearAndRemicadeSorted = hospitalDfYearSorted[hospitalDfYearSorted['Product_Remicade'] == True]
PharmacyDfYearAndRemicadeSorted = PharmacyDfYearSorted[PharmacyDfYearSorted['Product_Remicade'] == True]
sickHouseDfYearAndRemicadeSorted = sickHouseDfYearSorted[sickHouseDfYearSorted['Product_Remicade'] == True]
mergedHospitalPharmacySickHouseDfYearAndRemicadeSorted = pd.concat([hospitalDfYearAndRemicadeSorted, PharmacyDfYearAndRemicadeSorted, sickHouseDfYearAndRemicadeSorted], ignore_index=True)

print('Remicade sales:')
print('Hospital total sales btw 2010 and 2019 for Remicade: ', sum(hospitalDfYearAndRemicadeSorted['Volume']))
print('Pharmacy total sales btw 2010 and 2019 for Remicade: ', sum(PharmacyDfYearAndRemicadeSorted['Volume']))
print('Sickhouse total sales btw 2010 and 2019 for Remicade: ', sum(sickHouseDfYearAndRemicadeSorted['Volume']))
print()

#Remsima
hospitalDfYearAndRemsimaSorted = hospitalDfYearSorted[hospitalDfYearSorted['Product_Remsima'] == True]
PharmacyDfYearAndRemsimaSorted = PharmacyDfYearSorted[PharmacyDfYearSorted['Product_Remsima'] == True]
sickHouseDfYearAndRemsimaSorted = sickHouseDfYearSorted[sickHouseDfYearSorted['Product_Remsima'] == True]
mergedHospitalPharmacySickHouseDfYearAndRemsimaSorted = pd.concat([hospitalDfYearAndRemsimaSorted, PharmacyDfYearAndRemsimaSorted, sickHouseDfYearAndRemsimaSorted], ignore_index=True)

print('Remsima sales:')
print('Hospital total sales btw 2010 and 2019 for Remsima: ', sum(hospitalDfYearAndRemsimaSorted['Volume']))
print('Pharmacy total sales btw 2010 and 2019 for Remsima: ', sum(PharmacyDfYearAndRemsimaSorted['Volume']))
print('Sickhouse total sales btw 2010 and 2019 for Remsima: ', sum(sickHouseDfYearAndRemsimaSorted['Volume']))
print()

#Zessly
hospitalDfYearAndZesslySorted = hospitalDfYearSorted[hospitalDfYearSorted['Product_Zessly'] == True]
PharmacyDfYearAndZesslySorted = PharmacyDfYearSorted[PharmacyDfYearSorted['Product_Zessly'] == True]
sickHouseDfYearAndZesslySorted = sickHouseDfYearSorted[sickHouseDfYearSorted['Product_Zessly'] == True]
mergedHospitalPharmacySickHouseDfYearAndZesslySorted = pd.concat([hospitalDfYearAndZesslySorted, PharmacyDfYearAndZesslySorted, sickHouseDfYearAndZesslySorted], ignore_index=True)

print('Zessly sales:')
print('Hospital total sales btw 2010 and 2019 for Zessly: ', sum(hospitalDfYearAndZesslySorted['Volume']))
print('Pharmacy total sales btw 2010 and 2019 for Zessly: ', sum(PharmacyDfYearAndZesslySorted['Volume']))
print('Sickhouse total sales btw 2010 and 2019 for Zessly: ', sum(sickHouseDfYearAndZesslySorted['Volume']))

# %% [markdown]
# #### sort data into ATC5 code L04AC05

# %%
DfYearAndL04AC05SortedZessly = mergedHospitalPharmacySickHouseDfYearAndZesslySorted
DfYearAndL04AC05SortedRemsima = mergedHospitalPharmacySickHouseDfYearAndRemsimaSorted
DfYearAndL04AC05SortedRemicade = mergedHospitalPharmacySickHouseDfYearAndRemicadeSorted
DfYearAndL04AC05SortedInflectra = mergedHospitalPharmacySickHouseDfYearAndInflectraSorted

mergedDfYearAndL04AC05Sorted = pd.concat([DfYearAndL04AC05SortedZessly, DfYearAndL04AC05SortedRemsima, DfYearAndL04AC05SortedRemicade, DfYearAndL04AC05SortedInflectra], ignore_index=True)

# %% [markdown]
# #### Add type column to mergedHospitalPharmacySickHouseDfYearSorted

# %%
hospitalDfYearSorted['Type'] = 'Hospital'
sickHouseDfYearSorted['Type'] = 'Sickhouse'
PharmacyDfYearSorted['Type'] = 'Pharmacy'
mergedHospitalPharmacySickHouseDfYearAndSortedWithTypes = pd.concat([hospitalDfYearSorted, sickHouseDfYearSorted, PharmacyDfYearSorted], ignore_index=True)

# Add the 'Type' column to the 2nd column
cols = mergedHospitalPharmacySickHouseDfYearAndSortedWithTypes.columns.tolist()
cols = cols[:1] + cols[-1:] + cols[1:-1]
mergedHospitalPharmacySickHouseDfYearAndSortedWithTypes = mergedHospitalPharmacySickHouseDfYearAndSortedWithTypes[cols]

# %% [markdown]
# #### Convert all string and boolean values to numericals

# %%
# convert all unique values in 'Type' to numbers
mergedHospitalPharmacySickHouseDfYearAndSortedWithTypes['Type'] = pd.factorize(mergedHospitalPharmacySickHouseDfYearAndSortedWithTypes['Type'])[0]

# convert all unique values in 'WHO ATC 5 Code' to numbers
mergedHospitalPharmacySickHouseDfYearAndSortedWithTypes['WHO ATC 5 Code'] = pd.factorize(mergedHospitalPharmacySickHouseDfYearAndSortedWithTypes['WHO ATC 5 Code'])[0]

# convert all true false values to 1 and 0
mergedHospitalPharmacySickHouseDfYearAndSortedWithTypes['Region_Hovedstaden'] = mergedHospitalPharmacySickHouseDfYearAndSortedWithTypes['Region_Hovedstaden'].astype(int)
mergedHospitalPharmacySickHouseDfYearAndSortedWithTypes['Region_Midtjylland'] = mergedHospitalPharmacySickHouseDfYearAndSortedWithTypes['Region_Midtjylland'].astype(int)
mergedHospitalPharmacySickHouseDfYearAndSortedWithTypes['Region_Nordjylland'] = mergedHospitalPharmacySickHouseDfYearAndSortedWithTypes['Region_Nordjylland'].astype(int)
mergedHospitalPharmacySickHouseDfYearAndSortedWithTypes['Region_Sjælland'] = mergedHospitalPharmacySickHouseDfYearAndSortedWithTypes['Region_Sjælland'].astype(int)
mergedHospitalPharmacySickHouseDfYearAndSortedWithTypes['Region_Syddanmark'] = mergedHospitalPharmacySickHouseDfYearAndSortedWithTypes['Region_Syddanmark'].astype(int)
mergedHospitalPharmacySickHouseDfYearAndSortedWithTypes['Product_Cimzia'] = mergedHospitalPharmacySickHouseDfYearAndSortedWithTypes['Product_Cimzia'].astype(int)
mergedHospitalPharmacySickHouseDfYearAndSortedWithTypes['Product_Inflectra'] = mergedHospitalPharmacySickHouseDfYearAndSortedWithTypes['Product_Inflectra'].astype(int)
mergedHospitalPharmacySickHouseDfYearAndSortedWithTypes['Product_Remicade'] = mergedHospitalPharmacySickHouseDfYearAndSortedWithTypes['Product_Remicade'].astype(int)
mergedHospitalPharmacySickHouseDfYearAndSortedWithTypes['Product_Remsima'] = mergedHospitalPharmacySickHouseDfYearAndSortedWithTypes['Product_Remsima'].astype(int)
mergedHospitalPharmacySickHouseDfYearAndSortedWithTypes['Product_Stelara'] = mergedHospitalPharmacySickHouseDfYearAndSortedWithTypes['Product_Stelara'].astype(int)
mergedHospitalPharmacySickHouseDfYearAndSortedWithTypes['Product_Zessly'] = mergedHospitalPharmacySickHouseDfYearAndSortedWithTypes['Product_Zessly'].astype(int)

# %% [markdown]
# #### Define a dataset with data only from Central Apoteket (main pharmacy next to central station)

# %%
centralPharmacy = PharmacyDfYearAndStelaraSorted[PharmacyDfYearAndStelaraSorted['Account Description'].str.contains('CentralApoteket', case=False)]

# %% [markdown]
# #### Draw stacked column chart of total sales of Stelara in hospitals and pharmacies btw 2010 and 2019

# %%
hospitalDfYearAndStelaraSorted['Type'] = 'Hospitals'
sickHouseDfYearAndStelaraSorted['Type'] = 'Sickhouses'
PharmacyDfYearAndStelaraSorted['Type'] = 'Pharmacies'
mergedHospitalPharmacySickHouseDfYearAndStelaraSortedWithTypes = pd.concat([hospitalDfYearAndStelaraSorted, PharmacyDfYearAndStelaraSorted, sickHouseDfYearAndStelaraSorted], ignore_index=True)

# Pivot the data
pivot_df_stelara = mergedHospitalPharmacySickHouseDfYearAndStelaraSortedWithTypes.pivot_table(index='Year Month (after 2000) in Datetime', columns='Type', values='Volume', aggfunc='sum').fillna(0)

# %% [markdown]
# #### Draw stacked column chart of total sales of Cimzia in hospitals and pharmacies btw 2010 and 2019

# %%
hospitalDfYearAndCimziaSorted['Type'] = 'Hospitals'
sickHouseDfYearAndCimziaSorted['Type'] = 'Sickhouses'
PharmacyDfYearAndCimziaSorted['Type'] = 'Pharmacies'
mergedHospitalPharmacySickHousesDfYearAndCimziaSortedWithTypes = pd.concat([hospitalDfYearAndCimziaSorted, PharmacyDfYearAndCimziaSorted, sickHouseDfYearAndCimziaSorted], ignore_index=True)

# Pivot the data
pivot_df_cimzia = mergedHospitalPharmacySickHousesDfYearAndCimziaSortedWithTypes.pivot_table(index='Year Month (after 2000)', columns='Type', values='Volume', aggfunc='sum').fillna(0)

# %% [markdown]
# #### Draw stacked column chart of total sales of Inflectra in hospitals and pharmacies btw 2010 and 2019

# %%
hospitalDfYearAndInflectraSorted['Type'] = 'Hospitals'
sickHouseDfYearAndInflectraSorted['Type'] = 'Sickhouses'
PharmacyDfYearAndInflectraSorted['Type'] = 'Pharmacies'
mergedHospitalPharmacySickHouseDfYearAndInflectraSortedWithTypes = pd.concat([hospitalDfYearAndInflectraSorted, PharmacyDfYearAndInflectraSorted, sickHouseDfYearAndInflectraSorted], ignore_index=True)

# Pivot the data
pivot_df_inflectra = mergedHospitalPharmacySickHouseDfYearAndInflectraSortedWithTypes.pivot_table(index='Year Month (after 2000)', columns='Type', values='Volume', aggfunc='sum').fillna(0)

# %% [markdown]
# #### Draw stacked column chart of total sales of Remicade in hospitals and pharmacies btw 2010 and 2019

# %%
hospitalDfYearAndRemicadeSorted['Type'] = 'Hospitals'
sickHouseDfYearAndRemicadeSorted['Type'] = 'Sickhouses'
PharmacyDfYearAndRemicadeSorted['Type'] = 'Pharmacies'
mergedHospitalPharmacySickHouseDfYearAndRemicadeSortedWithTypes = pd.concat([hospitalDfYearAndRemicadeSorted, PharmacyDfYearAndRemicadeSorted, sickHouseDfYearAndRemicadeSorted], ignore_index=True)

# Pivot the data
pivot_df_remicade = mergedHospitalPharmacySickHouseDfYearAndRemicadeSortedWithTypes.pivot_table(index='Year Month (after 2000)', columns='Type', values='Volume', aggfunc='sum').fillna(0)

# %% [markdown]
# #### Draw stacked column chart of total sales of Remsima in hospitals and pharmacies btw 2010 and 2019

# %%
hospitalDfYearAndRemsimaSorted['Type'] = 'Hospitals'
sickHouseDfYearAndRemsimaSorted['Type'] = 'Sickhouses'
PharmacyDfYearAndRemsimaSorted['Type'] = 'Pharmacies'
mergedHospitalPharmacySickHouseDfYearAndRemsimaSortedWithTypes = pd.concat([hospitalDfYearAndRemsimaSorted, PharmacyDfYearAndRemsimaSorted, sickHouseDfYearAndRemsimaSorted], ignore_index=True)

# Pivot the data
pivot_df_remsima = mergedHospitalPharmacySickHouseDfYearAndRemsimaSortedWithTypes.pivot_table(index='Year Month (after 2000)', columns='Type', values='Volume', aggfunc='sum').fillna(0)

# %% [markdown]
# #### Draw stacked column chart of total sales of Zessly in hospitals and pharmacies btw 2010 and 2019

# %%
hospitalDfYearAndZesslySorted['Type'] = 'Hospitals'
sickHouseDfYearAndZesslySorted['Type'] = 'Sickhouses'
PharmacyDfYearAndZesslySorted['Type'] = 'Pharmacies'
mergedHospitalPharmacySickHouseDfYearAndZesslySortedWithTypes = pd.concat([hospitalDfYearAndZesslySorted, PharmacyDfYearAndZesslySorted, sickHouseDfYearAndZesslySorted], ignore_index=True)

# Pivot the data
pivot_df_zessly = mergedHospitalPharmacySickHouseDfYearAndZesslySortedWithTypes.pivot_table(index='Year Month (after 2000)', columns='Type', values='Volume', aggfunc='sum').fillna(0)

# %% [markdown]
# #### Draw stacked column chart of total sales of L04AC05 in hospitals and pharmacies btw 2010 and 2019

# %%
DfYearAndL04AC05SortedZessly['Type'] = 'Zessly'
DfYearAndL04AC05SortedRemsima['Type'] = 'Remsima'
DfYearAndL04AC05SortedRemicade['Type'] = 'Remicade'
DfYearAndL04AC05SortedInflectra['Type'] = 'Inflectra'
DfYearAndL04AC05SortedWithTypes = pd.concat([DfYearAndL04AC05SortedZessly, DfYearAndL04AC05SortedInflectra, DfYearAndL04AC05SortedRemsima, DfYearAndL04AC05SortedRemicade], ignore_index=True)

# %% [markdown]
# #### Draw scatter, line, bubble, or combo plot to show relationship between sold stelara over time between 2010 and 2019 in central pharmacy

# %%
# Group the data by 'Year Month (after 2000)' and calculate the median of 'Value'
meanValaues = centralPharmacy.groupby('Year Month (after 2000) in Datetime')['Volume'].mean().round().reset_index()

# convert 'Year Month (after 2000)' and median_values to int
print(meanValaues)

# %% [markdown]
# #### Show correlation between Volume and other feeatures

# %%
m = hospitalPharmacySickHouseDf.drop('Account Description', axis=1)
m2 = m.drop('Size', axis=1)
m2 = m2.drop('Year Month (after 2000) in Datetime', axis=1)

atcs = m2['WHO ATC 5 Code'].unique()

# Convert each unique value in 'WHO ATC 5 Code' to an int between 0 and 2
m2['WHO ATC 5 Code'] = m2['WHO ATC 5 Code'].apply(lambda x: np.where(atcs == x)[0][0])

# Also convert all true false values to 1 and 0 in these columns: Region_Hovedstaden	Region_Midtjylland	Region_Nordjylland	Region_Sjælland	Region_Syddanmark	Product_Cimzia	Product_Inflectra	Product_Remicade	Product_Remsima	Product_Stelara	Product_Zessly
m2['Region_Hovedstaden'] = m2['Region_Hovedstaden'].astype(int)
m2['Region_Midtjylland'] = m2['Region_Midtjylland'].astype(int)
m2['Region_Nordjylland'] = m2['Region_Nordjylland'].astype(int)
m2['Region_Sjælland'] = m2['Region_Sjælland'].astype(int)
m2['Region_Syddanmark'] = m2['Region_Syddanmark'].astype(int)
m2['Product_Cimzia'] = m2['Product_Cimzia'].astype(int)
m2['Product_Inflectra'] = m2['Product_Inflectra'].astype(int)
m2['Product_Remicade'] = m2['Product_Remicade'].astype(int)
m2['Product_Remsima'] = m2['Product_Remsima'].astype(int)
m2['Product_Stelara'] = m2['Product_Stelara'].astype(int)
m2['Product_Zessly'] = m2['Product_Zessly'].astype(int)

m2

print(len(m2.columns))


