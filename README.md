# Ames Housing Price Prediction

This project explores and models the Ames Housing dataset to predict residential property prices. The goal is to build a robust pipeline that combines data preprocessing and model evaluation to accurately forecast sale prices.


## Key Objectives

- Clean and preprocess housing dataset.
- Understand patterns through exploratory data analysis.
- Compare multiple regression models i.e Linear Regression, Random Forest, and XGBoost.
- Evaluate model performance using RMSE and R² on log-transformed targets.


##  Workflow Breakdown

### 1. Data Loading & Setup
- Imported necessary libraries (pandas, numpy, seaborn, scikit-learn, XGBoost).
- Loaded AmesHousing.csv dataset.

### 2. Initial Inspection
- Explored dataset structure, checked datatypes, duplicates, and missing values.
- Visualized missing data using a barplot.

### 3. Data Cleaning
- Imputed numeric columns with median and categorical columns with mode.
- Verified that all missing values were handled appropriately.

### 4. Exploratory Data Analysis (EDA)
- Visualized the  target 'SalePrice' distribution.
- Generated correlation heatmaps to find highly correlated predictors.
- Explored relationships between SalePrice and top predictors via boxplots 
- Used lmplot to visualize pairwise relationships across all numeric features against SalePrice.

### 5. Preprocessing & Feature Engineering
- Applied log-transform to the target variable to reduce right skewness.
- Used ColumnTransformer to:
  - Standardize numeric features with StandardScaler
  - One-hot encode categorical features using OneHotEncoder

### 6. Model Training & Evaluation

Three regression models were trained and evaluated using scikit-learn pipelines: **Linear Regression**, **Random Forest**, and **XGBoost**. For all models, the same preprocessing strategy was applied—numeric features were scaled using StandardScaler, and categorical features were one-hot encoded using OneHotEncoder within a ColumnTransformer.

To address skewness in the target variable, a log transformation  was applied during training. After making predictions, the outputs were inverse-transformed to bring them back to their original scale for evaluation.

Performance was assessed using two metrics: **Root Mean Squared Error** and **R-squared**. Linear Regression served as a strong baseline model. Random Forest improved upon it by reducing prediction error and capturing non-linear relationships. XGBoost performed the best overall, achieving the lowest RMSE and the highest R- square around 0.92—indicating strong predictive accuracy.


##  Results Snapshot

- **Linear Regression**: RMSE ~ moderate, R-square  ~ good baseline  
- **Random Forest**: Improved performance with lower error and higher R-square 
- **XGBoost**: Best performance with R-square ≈ 0.92, lowest RMSE  


