# Pandas Data Analysis Report

## Dataset Overview

The Titanic dataset contains passenger information such as age, gender, ticket class, fare, and survival status.

## Data Quality Issues

* Age column contained 177 missing values.
* Cabin column contained 687 missing values.
* Embarked column contained 2 missing values.

## Data Cleaning

* Removed duplicate records.
* Filled missing Age values using the median.
* Filled missing Embarked values using the mode.

## Feature Engineering

Created a new column called **Age_Group** with the categories:

* Child
* Young Adult
* Adult
* Senior

## Exploratory Data Analysis

Performed analysis on:

* Gender distribution
* Age distribution
* Passenger class distribution

## Grouping and Aggregation

Calculated survival rates by gender.

### Survival Rate by Gender

* Female: 74.2%
* Male: 18.9%

## Key Insights

1. Female passengers had a significantly higher survival rate.
2. Male passengers had a much lower survival rate.
3. The dataset contains substantial missing Cabin information.
4. Most passengers belonged to the third passenger class.
5. Age required cleaning before analysis.
6. Passenger demographics influenced survival outcomes.
7. The dataset was successfully cleaned and transformed.
8. Visualizations helped identify important patterns.

## Conclusion

The Titanic dataset was cleaned, analyzed, and visualized using Pandas. Basic insights were generated through grouping, aggregation, and exploratory analysis.
