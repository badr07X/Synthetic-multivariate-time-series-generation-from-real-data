<p align="center">
  <img width="100px" height="100px" alt="ENSEA" src="https://www.ensea.fr/sites/all/themes/custom/customer/logo.png">
</p>

# Synthetic-multivariate-time-series-generation-from-real-data
The typical order of operations is imputation followed by generation, but this depends on the context of your task. Here's why:
1. Imputation First
Why: If your dataset has missing values, imputing these missing entries is necessary to ensure the data is complete before you proceed to any data generation process. Missing values could distort the patterns in your data, making the generation of new samples based on that data incorrect or unreliable.
When: When your dataset has gaps that need to be filled before further analysis or model training. For example, in time series data, imputing missing values can restore the continuity of the series, making it suitable for model generation or simulation.
2. Generation Second
Why: After imputation, you can then generate new samples or augment your dataset because your data is now complete. If generation is done before imputation, the model may end up creating unrealistic or inconsistent data due to the missing information.
When: After the missing values are imputed, generation can help to create new samples for training purposes, particularly if you're working with generative models like DMs or GANs.
In summary:
Imputation first ensures that your data is complete and consistent before generating new data points.
Generation second allows you to create new data based on the now-complete dataset.
In some cases, you may have to loop between the two processes (e.g., impute, generate, then re-impute if new missing data appears during generation), but the usual sequence is to impute missing data before generating new data.

