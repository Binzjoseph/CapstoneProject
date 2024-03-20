#Code for the Third Approach - Long Term Causal Analysis

# Libraries and reading the csv
!pip install -U -q PyDrive
!pip install linearmodels
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from linearmodels.panel import PanelOLS
import statsmodels.api as sm

#THIRD APPROACH STARTS HERE LONG-TERM ONE
broaderwindow = 'Third_Approach_data.csv'
broad_df_normal = pd.read_csv(broaderwindow)
broad_df_normal

# Filtering the dataset for significant changes in rank
broad_df = broad_df_normal[abs(broad_df_normal['Rank Diff']) >= 5]

#Actual logic, the order the columns is matched with the second approach
# Defining the correct order of the independent variables as specified
ordered_columns_diff = [
    'Female faculty (%) Diff',
    'Value for money rank Diff',
    'International course experience rank Diff',
    'International board (%) Diff',
    'Faculty with doctorates (%) Diff',
    'Nwemployedat3months Diff',
    'Women on board (%) Diff',
    'Weighted salary (US$) Diff',
    'Career progress rank Diff',
    'Female students (%) Diff',
    'Careers service rank Diff',
    'Salary percentage increase Diff',
    'International students (%) Diff',
    'International work mobility rank Diff',
    'International faculty (%) Diff',
    'Aims achieved (%) Diff',
    'Internships(%) Diff',
    'Avg_Course_Length(Months) Diff'
]
ordered_columns = [col.replace(' Diff', '') for col in ordered_columns_diff]
# Creating a single year identifier by taking the first year in the 'Year Diff' string
broad_df['Year'] = broad_df['Year Diff'].str.split('-').str[0].astype(int)
# Setting the index to be the entity identifier (School Name) and the new time identifier (Year)
broad_df = broad_df.set_index(['School Name', 'Year'])
broad_y = broad_df['Rank Diff']

# Dropping the 'Year Diff' column as we won't be using it in the model
broad_X = broad_df.drop(columns=['Rank Diff', 'Year Diff'])
# Reordering the columns based on the specified order with 'Diff'
broad_X = broad_X[ordered_columns_diff]
# Renaming the independent variables by removing ' Diff' and adding '2yr Change in' prefix
rename_mapping = {old: '2yr Change in ' + new for old, new in zip(ordered_columns_diff, ordered_columns)}
broad_X = broad_X.rename(columns=rename_mapping)
# Creating the fixed effects model for the significant changes dataset
model_significant = PanelOLS(broad_y, broad_X, entity_effects=True)
# Fitting the model
fitted_model_significant = model_significant.fit(cov_type='clustered', cluster_entity=True)
# Output of the summary of the model for significant changes
print(fitted_model_significant.summary)

# Sensitivity Analysis
# Defining a list of the predictors
all_predictors = [
    'Aims achieved (%) Diff', 'Avg_Course_Length(Months) Diff', 'Career progress rank Diff',
    'Careers service rank Diff', 'Faculty with doctorates (%) Diff', 'Female faculty (%) Diff',
    'Female students (%) Diff', 'International board (%) Diff', 'International course experience rank Diff',
    'International faculty (%) Diff', 'International students (%) Diff', 'International work mobility rank Diff',
    'Internships(%) Diff', 'Nwemployedat3months Diff', 'Salary percentage increase Diff',
    'Value for money rank Diff', 'Weighted salary (US$) Diff', 'Women on board (%) Diff'
]

# Defining the function to run the fixed effects model
def run_fixed_effects(broad_y, broad_X, entity_effects=True, cov_type='clustered', cluster_entity=True):
    model = PanelOLS(broad_y, broad_X, entity_effects=entity_effects)
    fitted_model = model.fit(cov_type=cov_type, cluster_entity=cluster_entity)
    return fitted_model

# Running the sensitivity analysis by excluding predictors one at a time based on p-values
for predictor in all_predictors:
    print(f"Running model without {predictor}")
    X_sensitivity = broad_df.drop(columns=[predictor, 'Rank Diff'])
    y_sensitivity = broad_df['Rank Diff']
    model_sensitivity = run_fixed_effects(y_sensitivity, X_sensitivity)
    print(model_sensitivity.summary)

# Converting the significant results into a dataframe
# Here we are using for saving the data frame
significant_factors = fitted_model_significant.summary.tables[1]  # Table 1 usually contains the results
significant_factors_df = pd.DataFrame(significant_factors.data[1:], columns=significant_factors.data[0])
# Converting numerical columns from string to float
significant_factors_df[['Parameter', 'Std. Err.', 'T-stat', 'P-value', 'Lower CI', 'Upper CI']] = significant_factors_df[['Parameter', 'Std. Err.', 'T-stat', 'P-value', 'Lower CI', 'Upper CI']].astype(float)
# Filtering only significant factors
significant_threshold = 0.05
significant_factors_df = significant_factors_df[significant_factors_df['P-value'] < significant_threshold]
significant_factors_df

# Saving the DataFrame to Excel
# Saving the DataFrame to an Excel file
significant_factors_df.to_excel('significant_factors.xlsx', index=False)
# Downloading the excel file
from google.colab import files

# Saving the excel file
files.download('significant_factors.xlsx')
