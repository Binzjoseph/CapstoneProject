#Code for the First Approach  - Normal Prediction Model

# Loading the libraries the first five are for Google Colab specifically
!pip install -U -q PyDrive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

#For reading the csv first 5 is for google colab
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)
file_id = 'First_Approach_data.csv'
df = pd.read_csv(file_id)
df

# Defining the dependent variable (y) and independent variables (X)
y = df['Rank']
y
X = df[['Female faculty (%)','Value for money rank','International course experience rank','International board (%)','Faculty with doctorates (%)','Nwemployedat3months','Women on board (%)','Weighted salary (US$)','Career progress rank','Female students (%)','Careers service rank','Salary percentage increase','International students (%)','International work mobility rank','International faculty (%)','Aims achieved (%)','Internships(%)','Avg_Course_Length(Months)']]
X
# Calculating the correlation matrix
correlation_matrix = X.corrwith(y)
print(correlation_matrix)

#Train test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Multicollinearity Check
#VIF multicollinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor
#VIF for each feature
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
#VIF values
print(vif.round(1))

# Models for the First Approach only the main ones are included
#Linear Regression
model = linear_model.LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared (R2): {r2}")

# Random Forest Regression
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred_rf)
r2 = r2_score(y_test, y_pred_rf)
print(f"Mean Squared Error: {mse}")
print(f"R-squared (R2): {r2}")
#for feature importance in Random Forest
rf_coeffs = dict(zip(X.columns, rf_model.feature_importances_))
print(rf_coeffs)

# ElasticNet Regression
elasticnet_model = ElasticNet(alpha=1.0, l1_ratio=0.5)
elasticnet_model.fit(X_train, y_train)
y_pred_elasticnet = elasticnet_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred_elasticnet)
r2 = r2_score(y_test, y_pred_elasticnet)
print(f"Mean Squared Error: {mse}")
print(f"R-squared (R2): {r2}")

# Bayesian Linear Regression
bayesian_model = BayesianRidge()
bayesian_model.fit(X_train, y_train)
y_pred_bayesian = bayesian_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred_bayesian)
r2 = r2_score(y_test, y_pred_bayesian)
print(f"Mean Squared Error: {mse}")
print(f"R-squared (R2): {r2}")

#Coefficients
bayesian_coeffs = dict(zip(X.columns, bayesian_model.coef_))
print(bayesian_coeffs)
# Convert the dictionary to a DataFrame
bayesian_coeffs_df = pd.DataFrame(bayesian_coeffs.items(), columns=['Column Name', 'Coefficient'])
# Display the DataFrame
print(bayesian_coeffs_df)
