# DeepRetail
<img src="https://img.shields.io/badge/Maintained%20by-Vives%20AI%20Lab-red"> [![Downloads](https://static.pepy.tech/personalized-badge/DeepRetail?period=total&units=international_system&left_color=grey&right_color=blue&left_text=downloads)](https://pepy.tech/project/DeepRetail) <img src="https://img.shields.io/badge/python-v3.7%2B-blue"> <img src="https://img.shields.io/badge/pypi-v0.0.7-blue">

Python package on deep learning AI and machine learning for Retail

This package is developed by the AI team at [VIVES University of Applied Sciences](https://www.vives.be/en/research/centre-expertise-business-management) and is used in our research on [demand forecasting](https://yvesrsagaert.wordpress.com/).

___


## Getting started

### Installation

1. Install python3.7+
2. Create a virtual env where you want to install:

    ```
    $> python3 -m venv retailanalytics
    ```

3. Activate the environment

    ```
    $> source retailanalytics/bin/activate
    ```

4. Install the package with pip

     ```
    $> pip install DeepRetail
     ```
	 
### Use hierarchical modelling
```python
import pandas as pd
from DeepRetail.transformations.formats import transaction_df
from DeepRetail.forecasting.statistical import StatisticalForecaster

# Load
df = pd.read_csv('daily_data.csv', index_col=0)

# Get a sample 
sampled_df = df.sample(20)

# Convert to transaction
t_df = transaction_df(sampled_df)

# Define the parameters
freq = 'M'
h = 4
holdout = True
cv = 2
models = ['ETS', 'Naive']

# Convert columns to datetime
sampled_df.columns = pd.to_datetime(sampled_df.columns)

# Resample columns to montly frequency
sampled_df = sampled_df.resample('M', axis=1).sum()

# Define the forecaster
forecaster = StatisticalForecaster(models = models, freq = freq)

# Fit the forecaster
forecaster.fit(sampled_df, format = 'pivoted')

# Predict
forecast_df = forecaster.predict(h = h, cv = cv, holdout = holdout)
forecast_df.head()

```

## Contributing

Contribution is welcomed! 

Start by reviewing the [contribution guidelines](https://github.com/yForecasting/DeepRetail/blob/main/CONTRIBUTING.md). After that, take a look at a [good first issue](https://github.com/yForecasting/DeepRetail/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).


## Disclaimer
`DeepRetail` is an open-source package. We do our best to make this package robust and stable, but we do not take liability for any errors or instability. 

## Support

The [AI team](https://yforecasting.github.io/) at VIVES University of Applied Sciences builds and maintains `DeepRetail` to make it simple and accessible. We are using this software in our research on [demand forecasting](https://yvesrsagaert.wordpress.com/). A special thanks to Ruben Vanhecke and Filotas Theodosiou for their contribution. The [maintenance workflow](https://github.com/yForecasting/DeepRetail/blob/main/MAINTAINING.md) can be found here.