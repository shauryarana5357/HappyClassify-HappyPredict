# HappyClassify — World Happiness ML Suite

Machine learning classification and regression project built on the World Happiness Report dataset, developed as part of the Summer Internship Programme 2026 at the Department of Computer Science & Engineering, Dr. B. R. Ambedkar National Institute of Technology Jalandhar.
The purpose of this project is to build, compare, and deploy multiple machine learning models that classify countries into happiness levels and predict happiness scores, and to present the results through an interactive web dashboard.
Should the person who is looking at this analysis have any questions or suggestions, do not hesitate to contact me.

## The Data

Our dataset is the World Happiness Report, published annually by the United Nations Sustainable Development Solutions Network, covering the years 2005–2023 (~2,364 country-year observations). The report ranks countries by their overall happiness (Life Ladder score), derived from Gallup World Poll surveys across more than 160 countries.

Each country-year observation includes the following socio-economic indicators:

- Log GDP per capita
- Social support
- Healthy life expectancy at birth
- Freedom to make life choices
- Generosity
- Perceptions of corruption
- Positive affect
- Negative affect
- Dystopia residual (benchmark component)

From the raw Life Ladder score, countries are additionally grouped into four happiness levels: Unhappy, Moderately Happy, Happy, and Very Happy.

## Tasks performed

### A. Importing, cleaning and numerical summaries
1. Import the raw World Happiness Report dataset as a pandas DataFrame.
2. Check the number of observations and columns.
3. Obtain the column headings and data types for each column.
4. Check for missing values and generate a missing-values report.
5. Handle missing values, remove duplicates, and detect/treat outliers.
6. Encode categorical variables and scale numerical features (StandardScaler).
7. Split the cleaned dataset into train / validation / test sets (70% / 15% / 15%).
8. Obtain the mean, minimum, and maximum value for each numerical column.
9. List the top 10 happiest and 10 least happy countries.

### B. Indexing and grouping
1. Use the 'Region' column (derived via country-to-continent mapping) to group countries by region.
2. Compute the mean happiness score for each region and rank regions from most to least happy.
3. Compute the number of countries per region with a happiness score above 6.0.
4. Compute the happiness score range (max − min) for each region.

### C. Exploratory Data Analysis (visualizations)
1. Histogram of the Happiness Score distribution.
2. Box plot of happiness scores by region.
3. Correlation heatmap across all socio-economic indicators.
4. Count plot of countries by happiness level.
5. Pair plot of feature relationships.
6. Stacked bar plot of region vs. happiness level.
7. World choropleth map of average happiness score.
8. Radar chart comparing the top 5 happiest countries.
9. Violin plot of happiness score distribution by region.

### D. Feature Engineering
1. Check multicollinearity among features (VIF).
2. Engineer additional features where required.
3. Apply feature scaling (StandardScaler).
4. Perform feature selection via correlation analysis, recursive feature elimination (RFE), and model-based feature importance.

### E. Model Building & Comparison
1. Train multiple classification models — Logistic Regression, K-Nearest Neighbors, Decision Tree, Random Forest, and SVM — to predict happiness level (4-class target).
2. Train multiple regression models — Linear Regression, Decision Tree, SVR, and Random Forest — to predict the continuous Life Ladder score.
3. Evaluate classification models using Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
4. Evaluate regression models using MAE, MSE, RMSE, and R².
5. Identify the best-performing model for each task (Random Forest for both).

### F. Model Diagnostics
1. Confusion matrix per model.
2. ROC curves (one-vs-rest) per class.
3. Feature importance ranking.
4. Class distribution pie chart.
5. Learning curves (training vs. validation score).

### G. Interactive Prediction Dashboard
1. Build a Streamlit web app with two tools: HappyClassify (classification) and HappyPredict (regression).
2. Allow users to adjust socio-economic indicator sliders and get a live prediction with class probabilities / predicted score.
3. Support light/dark theme toggle and per-model selection for predictions.

## Running the app locally

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

## Results Summary

**Best classification model:** Random Forest — Accuracy 83.7%, ROC-AUC 0.954
**Best regression model:** Random Forest Regressor — R² 0.861, RMSE 0.407