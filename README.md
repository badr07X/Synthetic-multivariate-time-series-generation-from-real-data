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

3. Generation and Metadata

Role of Metadata: Metadata can play an even more significant role in data generation because generative models rely on the underlying patterns in the data to generate new samples. Metadata can provide constraints and information about the features being generated.

For example, metadata can help:

Conditional generation: Generate new samples conditioned on certain attributes or context (e.g., generating new data for a specific time period or category).

Feature dependencies: Ensure that the relationships between generated features are consistent with the original data.

Data structure: Define the dimensionality or feature set of the generated data to match the real data.

Use of Metadata in Generation: Generative models like GANs or DMs can be conditioned on metadata to create realistic samples. 

In time series generation, you might use timestamps as part of the metadata to create new time series data that follows the same temporal patterns.

CSDI : 



