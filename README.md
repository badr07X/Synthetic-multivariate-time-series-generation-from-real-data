<p align="center">
  <img width="100px" height="100px" alt="ENSEA" src="https://www.ensea.fr/sites/all/themes/custom/customer/logo.png">
</p>

# Synthetic-multivariate-time-series-generation-from-real-data
## Introduction  

   This project explores methods for generating synthetic multivariate time series that accurately capture real-world dynamics. It focuses on using deep learning, statistical models, and generative models (e.g., GANs and Diffusion Models ) to create high-fidelity synthetic datsets.

   These datasets can support tasks such as data analysis, model training, and validation, addressing issues of data privacy and scarcity. The project will also assess the quality of the generated data by comparing it to real-world data based on statistical properties and predictive performance.

### 1.1 Times series :  
   Time series data exhibit several distinct characteristics that define their behavior and guide their analysis:

   #### Trend
    A long-term upward or downward movement in the data over time, reflecting persistent changes such as growth, decline, or seasonal shifts.

   #### Seasonality
    Regular, periodic patterns or fluctuations within the data that occur at fixed intervals, such as daily, weekly, monthly, or yearly.
    Example: Increased retail sales during holiday seasons.

   #### Cyclicality
    Recurrent fluctuations that do not follow a fixed calendar period but are influenced by economic or business cycles. Cycles tend to be longer than seasonal patterns.

   #### Stationarity
    A stationary time series has constant mean, variance, and autocorrelation over time. Non-stationary series often need transformation (e.g., differencing) for effective analysis.

   #### Autocorrelation
    The correlation of the time series with its past values, indicating how current observations are influenced by previous ones.

   #### Noise
    Random or unexplained variability in the data, which is often non-systematic and does not follow identifiable patterns.

   #### Data Frequency
    The interval at which data points are collected (e.g., daily, hourly, annually) significantly affects the type of analysis and resolution of the insights.

   #### Sparsity or Irregularity
    Some time series may have missing or irregularly spaced data points, which requires specific handling techniques.

   #### Volatility
    Variability or dispersion of the data points, often present in financial or economic time series, affecting predictability and requiring specialized models.

   ####  Multivariate Relationships
    Some time series may interact with others (e.g., stock prices and interest rates), necessitating multivariate analysis.

#### Key Takeaway

Understanding these characteristics helps determine appropriate modeling techniques, such as ARIMA, exponential smoothing, or neural networks, and facilitates accurate forecasting and decision-making.



### 1.2 The utility of time series :
  Time series analysis allows for the study and prediction of dynamic phenomena, such as sales trends, financial markets, or energy demand.

  The generation of time series data is equally important, as it provides synthetic datasets for testing models, simulating scenarios, and training AI systems when real-world data is limited or incomplete. This capability enables the design of robust models and the exploration of hypothetical situations.

  Time series are crucial for forecasting, anomaly detection, and process optimization across a wide range of fields, including finance, healthcare, and environmental studies.

  Despite their complexity, advancements in artificial intelligence are enhancing their generation, analysis, and application, further strengthening their role in decision-making and innovation.
   
   
   
## Diffusion Models (DMs)


The typical order of operations is imputation followed by generation, but this depends on the context of your task. Here's why:

## 1. Imputation First
 
Why:  If your dataset has missing values, imputing these missing entries is necessary to ensure the data is complete before you proceed to any data generation process. Missing values could distort the patterns in your data, making the generation of new samples based on that data incorrect or unreliable.
When: When your dataset has gaps that need to be filled before further analysis or model training. For example, in time series data, imputing missing values can restore the continuity of the series, making it suitable for model generation or simulation.

## 2. Generation Second

Why: After imputation, you can then generate new samples or augment your dataset because your data is now complete. If generation is done before imputation, the model may end up creating unrealistic or inconsistent data due to the missing information.
When: After the missing values are imputed, generation can help to create new samples for training purposes, particularly if you're working with generative models like DMs or GANs.

In summary:

Imputation first ensures that your data is complete and consistent before generating new data points.
Generation second allows you to create new data based on the now-complete dataset.
In some cases, you may have to loop between the two processes (e.g., impute, generate, then re-impute if new missing data appears during generation), but the usual sequence is to impute missing data before generating new data.

## 3. Generation and Metadata


Role of Metadata: Metadata can play an even more significant role in data generation because generative models rely on the underlying patterns in the data to generate new samples. Metadata can provide constraints and information about the features being generated.

For example, metadata can help:

Conditional generation: Generate new samples conditioned on certain attributes or context (e.g., generating new data for a specific time period or category).

Feature dependencies: Ensure that the relationships between generated features are consistent with the original data.

Data structure: Define the dimensionality or feature set of the generated data to match the real data.

Use of Metadata in Generation: Generative models like GANs or DMs can be conditioned on metadata to create realistic samples. 

In time series generation, you might use timestamps as part of the metadata to create new time series data that follows the same temporal patterns.

## CSDI : 
<p align="center">
  <img alt="csdi model" src="https://github.com/badr07X/Synthetic-multivariate-time-series-generation-from-real-data/blob/main/figures/model.png">
</p>

  ### CSDI hyperparameters 
### Table 1: CSDI Hyperparameters

| **Hyperparameter**                | **Value**     |
|-----------------------------------|---------------|
| Residual layers                   | 4             |
| Residual channels                 | 64            |
| Diffusion embedding dim.          | 128           |
| Schedule                          | Quadratic     |
| Diffusion steps \(T\)             | 50            |
| \( \beta_0 \) (Start of variance) | 0.0001        |
| \( \beta_1 \) (End of variance)   | 0.5           |
| Feature embedding dim.            | 128           |
| Time embedding dim.               | 16            |
| Self-attention layers time dim.   | 1             |
| Self-attention heads time dim.    | 8             |
| Self-attention layers feature dim.| 1             |
| Self-attention heads feature dim. | 8             |
| Optimizer                         | Adam          |
| Loss function                     | MSE           |
| Learning rate                     | \(1 \times 10^{-3}\) |
| Weight decay                      | \(1 \times 10^{-6}\) |


## Time Weaver :  
<p align="center">
  <img alt=" csdi Time Weaver" src="https://github.com/badr07X/Synthetic-multivariate-time-series-generation-from-real-data/blob/main/figures/time%20Weaver%20.png">
</p>

## Time Weaver CSDI : 

<p align="center">
  <img alt=" csdi Time Weaver" src="https://github.com/badr07X/Synthetic-multivariate-time-series-generation-from-real-data/blob/main/figures/Time%20Weaver-CSDI.png">
</p>




