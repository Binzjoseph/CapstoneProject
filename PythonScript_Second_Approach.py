#Code for the Second Approach  - Short Term Causal Analysis
# Data Generation of second Approach
#First loading and reading the csv as done in the previous sections
!pip install -U -q PyDrive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
import pandas as pd
import pandas as pd
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)
file_id = 'First_Approach_data.csv'
df = pd.read_csv(file_id)
df

# Logic for the Data generation 
# Defining the years for which we want to calculate differences
years = [2019, 2020, 2021, 2022, 2023]
# Creating an empty DataFrame to store the differences
diff_df = pd.DataFrame()

# Calculating the differences for each attribute between consecutive years
attribute_columns = [
    'Rank',
    'Female faculty (%)',
    'Value for money rank',
    'International course experience rank',
    'International board (%)',
    'Faculty with doctorates (%)',
    'Nwemployedat3months',
    'Women on board (%)',
    'Weighted salary (US$)',
    'Career progress rank',
    'Female students (%)',
    'Careers service rank',
    'Salary percentage increase',
    'International students (%)',
    'International work mobility rank',
    'International faculty (%)',
    'Aims achieved (%)',
    'Internships(%)',
    'Avg_Course_Length(Months)'
]
for i in range(1, len(years)):
    current_year = years[i]
    previous_year = years[i - 1]
    # Filtering the data for the current and previous years
    current_data = df[df['Year'] == current_year]
    previous_data = df[df['Year'] == previous_year]

    # Calculating the differences for each attribute by subtracting the previous year's value
    year_difference = f'{current_year}-{previous_year}'
    diff = current_data.set_index('School Name')[attribute_columns] - previous_data.set_index('School Name')[attribute_columns]
    diff.reset_index(inplace=True)
    diff['Year'] = year_difference
    # Appending the results to the diff_df DataFrame
    diff_df = pd.concat([diff_df, diff])
print(diff_df)

#Saving the dataframe to excel format and downloading the excel in google colab
# Saving the results to an Excel file
diff_df.to_excel('Second_Approach_data.xlsx', index=False)
#Downloading
from google.colab import files
# Saving the excel file
files.download('attribute_differences.xlsx')


# The Actual Short-Term Causal Approach using the data generated above the earlier file was converted to csv before reading
#Importing Libraries and installing linear models first five is for google colab
!pip install -U -q PyDrive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
!pip install linearmodels
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from linearmodels.panel import PanelOLS
import statsmodels.api as sm

# Reading the csv file first five for google colab specific
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)
file_id = ‘Second_Approach_data.csv’
imm_df_n = pd.read_csv(file_id)
imm_df_n

#Actual Logic
# Renaming 'Rank diff' to 'Rank Diff'
imm_df_n.rename(columns={'Rank diff': 'Rank Diff'}, inplace=True)
# Renaming 'Year' to 'Year Diff'
imm_df_n.rename(columns={'Year': 'Year Diff'}, inplace=True)
imm_df_n

#Filtering for difference greater than or equal to 5
imm_df = imm_df_n[imm_df_n['Rank Diff'].abs() >= 5]
imm_df


# The Fixed Effect Regression Model
# Creating a single year identifier by taking the first year in the 'Year Diff' string
imm_df['Year'] = imm_df['Year Diff'].str.split('-').str[0].astype(int)
# Setting the index to be the entity identifier (School Name) and the new time identifier (Year)
imm_df = imm_df.set_index(['School Name', 'Year'])
# The dependent variable is 'Rank Diff' and the rest are independent variables
imm_df_y = imm_df['Rank Diff']
imm_df_X = imm_df.drop(columns=['Rank Diff', 'Year Diff'])  # Drop the original 'Year Diff' column
# Renaming the independent variables with '1yr Change in' prefix
imm_df_X = imm_df_X.rename(columns=lambda x: '1yr Change in ' + x)
# Creating the fixed effects model
model = PanelOLS(imm_df_y, imm_df_X, entity_effects=True)
# Fitting the model
fitted_model = model.fit(cov_type='clustered', cluster_entity=True)
# Output of the summary of the model
print(fitted_model.summary)

#Multicollinearity Check
#VIF multicollinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor
# Calculating VIF for each feature
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(imm_df_x.values, i) for i in range(imm_df_x.shape[1])]
vif["features"] = imm_df_x.columns
# Inspecting VIF values
print(vif.round(1))


# Converting the Significant Results into a DataFrame
table = fitted_model.summary.tables[1]
# Converting the table to a DataFrame
df = pd.DataFrame(table.data[1:], columns=table.data[0])
# Converting the numerical columns from string to float
df[['Parameter', 'Std. Err.', 'T-stat', 'P-value', 'Lower CI', 'Upper CI']] = df[['Parameter', 'Std. Err.', 'T-stat', 'P-value', 'Lower CI', 'Upper CI']].astype(float)
# Setting a threshold for significance, usually 0.05 or 0.01
significance_level = 0.05
# Filtering for significant factors only
significant_factors_df_short = df[df['P-value'] < significance_level]
significant_factors_df_short

#Exporting to Excel
# Exporting to Excel
significant_factors_df_short.to_excel('shorttermeffects.xlsx', index=False)
from google.colab import files
# Saving the excel file
files.download('shorttermeffects.xlsx')

#Filtering for difference greater than or equal to 10
imm_df_10 = imm_df[imm_df['Rank Diff'].abs() >= 10]
imm_df_10

#Model Building
# The dependent variable is 'Rank Diff' and the rest are independent variables
imm_df_10_y = imm_df_10['Rank Diff']
imm_df_10_X = imm_df_10.drop(columns=['Rank Diff', 'Year Diff'])  # Drop the original 'Year Diff' column
# Renaming the independent variables with '1yr Change in' prefix
imm_df_10_X = imm_df_10_X.rename(columns=lambda x: '1yr Change in ' + x)
# Creating the fixed effects model
model = PanelOLS(imm_df_10_y, imm_df_10_X, entity_effects=True)
# Fitting the model
fitted_model = model.fit(cov_type='clustered', cluster_entity=True)
# Output of the summary of the model
print(fitted_model.summary)
