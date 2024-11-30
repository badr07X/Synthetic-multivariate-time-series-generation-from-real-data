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


Diffusion models (DMs) are a type of generative model that learn to generate data by reversing a diffusion process. Below, we explain the **forward process**, **backward process**, and the corresponding **loss function**.

---

### 1. Forward Process

The forward process is a Markovian process that gradually adds noise to the data over a series of diffusion steps $T$. At each step $t$, noise is added to the data $x_0$ according to a predefined noise schedule $\{\beta_1, \beta_2, \dots, \beta_T\}$. 

The forward process is defined as:

$$
q(x_1, x_2, \dots, x_T \mid x_0) = \prod_{t=1}^T q(x_t \mid x_{t-1}),
$$

where:

$$
q(x_t \mid x_{t-1}) = \mathcal{N}(\sqrt{1 - \beta_t} x_{t-1}, \beta_t \mathbf{I}).
$$

- $x_0 \sim \mathcal{X}$: Initial data sample.
- $T$: Total number of diffusion steps.
- $q(x_T) \approx \mathcal{N}(0, \mathbf{I})$: After $T$ steps, the data becomes approximately Gaussian.

---

### 2. Backward Process

The backward process is trained to denoise the noisy data $x_T \sim \mathcal{N}(0, \mathbf{I})$ back to $x_0$. This is modeled as:

$$
p_\theta(x_0, x_1, \dots, x_{T-1} \mid x_T) = p(x_T) \prod_{t=1}^T p_\theta(x_{t-1} \mid x_t),
$$

where:

- $p(x_T) = \mathcal{N}(0, \mathbf{I})$: The prior distribution at $x_T$.
- $p_\theta(x_{t-1} \mid x_t)$: Parameterized by a neural network to approximate $q(x_{t-1} \mid x_t, x_0)$.

---

### 3. Training Objective (Loss Function)
(Loss Function)

To train the model, we optimize the network to predict the noise $\epsilon$ that was added at each step $t$. The loss function is:

![Loss Function](path_to_image/loss_function.png)

- $x_t$ is generated by adding noise $\epsilon$ to $x_0$ using the forward process.
- $\theta_{\text{denoiser}}$: A neural network trained to predict $\epsilon$ given $x_t$ and $t$.

This loss function is equivalent to **score-matching techniques**.



# Objectives : 

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

## Time Series Diffusion Model Architecture

This repository implements a diffusion-based model for time series generation and imputation. Below is a detailed explanation of the architecture.

## Overview
The model is designed for tasks like:
- **Time-series generation**: Synthesizing realistic data.
- **Imputation**: Filling in missing values.
- **Forecasting**: Predicting future time steps.

The architecture combines convolutional layers, transformers, and gated mechanisms to process temporal and feature dimensions effectively.

---

## Model Inputs
1. **Input Tensor**:  
   $x^{\text{co}}_0, x^{\text{ta}}_t$ of shape $(K, L, 2)$:  
   - $K$: Number of features.  
   - $L$: Length of the time series.  
   - $2$: Observed and target data.

2. **Diffusion Timestep Embedding**:  
   Encodes the diffusion timestep $t$ with shape $(1, 1, 128)$.

3. **Side Information**:
   - **Time Embedding**: $(1, L, 128)$.
   - **Feature Embedding**: $(K, 1, 16)$.
   - **Conditional Mask**: $m^{\text{co}}$ of shape $(K, L, 1)$, indicating observed vs. missing values.

---

## Model Components

### 1. Preprocessing Layers
- **Conv1x1 and ReLU**:  
  A 1D convolution followed by ReLU transforms the input tensor. Output shape: $(K, L, C)$.

- **Diffusion Embedding**:  
  The timestep embedding $t$ is processed with:
  - Fully-connected layers with SiLU activation.
  - Expansion via a convolution to shape $(1, L, 128)$.

- **Side Information Integration**:  
  The side information is expanded and concatenated with the input tensor along the feature dimension.

---

### 2. Transformer Layers
- **Temporal Transformer**:  
  Captures temporal dependencies across the time steps ($L$).  
  Input shape: $(K, L, C)$.

- **Feature Transformer**:  
  Models dependencies across features ($K$) at each time step.  
  Input shape: $(K, L, C)$.

---

### 3. Residual Blocks
Each residual block includes:
1. **Conv1x1**: Expands the feature channels to $2C$.
2. **Gated Activation Unit**: Combines input with a gating mechanism for nonlinear interactions.
3. **Conv1x1 (output)**: Reduces the channels back to $C$.  
   Residual connections preserve the original input and stabilize training.

---

### 4. Skip Connections
- Intermediate outputs are combined via skip connections to produce the final output.
- Ensures gradient flow and preserves fine-grained information.

---

### 5. Output Layer
1. **Conv1x1 and ReLU**: Final feature transformation.
2. **Masking**: Multiplies output by $(1 - m^{\text{co}})$ to mask observed indices.  
   Final output shape: $(K, L, 1)$.

---

## Key Features
1. **Temporal and Feature Separation**:  
   Uses separate transformers for time and feature dimensions.

2. **Diffusion Embedding**:  
   Ensures alignment with the diffusion framework by embedding the timestep.

3. **Side Information Utilization**:  
   Improves accuracy using auxiliary data like embeddings and masks.

4. **Residual and Skip Connections**:  
   Enhance stability, preserve input information, and allow deeper model training.

---

## Applications
This model can be used for:
- **Time-Series Generation**: Synthesizing realistic data for simulation or testing.
- **Imputation**: Filling in missing or corrupted time-series values.
- **Forecasting**: Predicting future values based on historical patterns.

For more details, refer to the code and implementation files in this repository.


  ### CSDI hyperparameters 
### Table 1: CSDI Hyperparameters

| **Hyperparameter**                | **Value**     |
|-----------------------------------|---------------|
| Residual layers                   | 4             |
| Residual channels                 | 64            |
| Diffusion embedding dim.          | 128           |
| Schedule                          | Quadratic     |
| Diffusion steps \(T\)             | 50            |
| $\( \beta_0 \)$ (Start of variance) | 0.0001        |
| $\( \beta_1 \)$ (End of variance)   | 0.5           |
| Feature embedding dim.            | 128           |
| Time embedding dim.               | 16            |
| Self-attention layers time dim.   | 1             |
| Self-attention heads time dim.    | 8             |
| Self-attention layers feature dim.| 1             |
| Self-attention heads feature dim. | 8             |
| Optimizer                         | Adam          |
| Loss function                     | MSE           |
| Learning rate                     | $\(1 \times 10^{-3}\)$ |
| Weight decay                      | $\(1 \times 10^{-6}\)$ |


## Time Weaver :  
<p align="center">
  <img alt=" csdi Time Weaver" src="https://github.com/badr07X/Synthetic-multivariate-time-series-generation-from-real-data/blob/main/figures/time%20Weaver%20.png">
</p>

### Time Weaver: A Conditional Time Series Generation Model

Time Weaver is a novel model designed for conditional time series generation through a diffusion-based probabilistic approach, enabling high-quality synthesis of time series data that aligns with provided metadata. The model incorporates a forward diffusion process $q(x_t \mid x_{t-1})$ to iteratively corrupt the input time series $x_0$ with Gaussian noise, producing increasingly noisy samples $x_t$ over $T$ steps. The reverse process aims to denoise $x_t$ step-by-step using a learnable denoising function $\theta_{\text{denoiser}}$, effectively reconstructing the original time series $x_0$.

The forward diffusion process can be mathematically represented as:

$$
q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{\alpha_t} x_{t-1}, (1 - \alpha_t) \mathbf{I}),
$$

where $\alpha_t$ defines the noise schedule over time $t$. The reverse process, parameterized by $\theta_{\text{denoiser}}$, predicts the distribution $p_\theta(x_{t-1} \mid x_t, z)$, conditioned on both the noisy sample $x_t$ and metadata embeddings $z$:

$$
p_\theta(x_{t-1} \mid x_t, z) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, z, t), \Sigma_\theta(x_t, z, t)).
$$

Here, $\mu_\theta$ and $\Sigma_\theta$ represent the predicted mean and variance, respectively, learned by the model.

---

### Metadata Integration

The key innovation of Time Weaver lies in its conditioning on both categorical and continuous metadata to guide the generation process. Metadata is processed through specialized tokenizers:

1. **Categorical metadata** $z_{\text{cat}}$ is transformed using a tokenizer $\theta_{\text{cat token}}$, which maps it into a learnable embedding space.
2. **Continuous metadata** $z_{\text{cont}}$ is processed by $\theta_{\text{cont token}}$, capturing continuous features in a similar embedding space.

These embeddings are concatenated:

$$
z = \text{Concat}(z_{\text{cat}}, z_{\text{cont}}),
$$

and passed through a self-attention mechanism parameterized by $\theta_{\text{condn}}$ to create the metadata embedding that conditions the denoising process. The attention mechanism ensures the model effectively integrates both types of metadata, producing:

$$
z_{\text{cond}} = \text{SelfAttention}(Q, K, V),
$$

where $Q, K, V$ represent the query, key, and value matrices derived from $z$.

---

### Training Objective
The model is trained to minimize the variational lower bound (VLB) of the data distribution:

![Loss Equation](https://latex.codecogs.com/svg.latex?\mathcal{L}=\mathbb{E}_{q(x_t%20\mid%20x_0)}%20\Big[%20%7C%20x_0%20-%20%5Chat{x}_0%20%7C%5E2%20%5D)

where $\hat{x}_0$ is the denoised prediction from $\theta_{\text{denoiser}}$. This objective encourages the model to accurately predict $x_0$ from noisy samples while aligning the generated data with the provided metadata.


### Iterative Sampling

During inference, Time Weaver begins with a Gaussian noise sample $x_T$ and iteratively refines it over $T$ steps using the reverse process:

$$
x_{t-1} = f_\theta(x_t, z, t) + g_\theta(x_t, z, t) \cdot \epsilon,
$$

where $f_\theta$ and $g_\theta$ define the denoising update rules, and $\epsilon \sim \mathcal{N}(0, \mathbf{I})$ is sampled noise.

---

### Summary

By combining diffusion models with metadata-aware conditioning, Time Weaver offers a powerful and flexible framework for generating realistic, contextually relevant time series data. Its applications span various fields, including healthcare, finance, and weather forecasting, where time series data is crucial.


## Time Weaver CSDI : 

<p align="center">
  <img alt=" csdi Time Weaver" src="https://github.com/badr07X/Synthetic-multivariate-time-series-generation-from-real-data/blob/main/figures/Time%20Weaver-CSDI.png">
</p>

## TimeWeaver: Conditional Score-based Diffusion Model for Time Series with Metadata Embedding

### Introduction

The **TimeWeaver** model extends the Conditional Score-based Diffusion Model (CSDI) by integrating **metadata embeddings**, allowing it to incorporate external contextual information for tasks such as time series imputation, forecasting, and conditional generation.

---

### Architecture Overview

The TimeWeaver architecture introduces:
1. A **metadata embedding block** to encode external information (e.g., weather, time of day).
2. Enhanced modeling of temporal and feature dependencies using dual transformer layers.

Below is the detailed explanation of the components in the architecture:

---

### Input Representations

### 1. Main Input
- **Input Tensor**: $x$ of shape $(F, 1, L)$  
  - $F$: Number of features.  
  - $L$: Sequence length (time steps).  

  The input passes through a **Conv1D** layer, producing a latent representation of shape $(F, C, L)$, where $C$ is the latent channel size.

- **Permutation**: The latent representation is permuted to align with the metadata embedding.

---

### 2. Metadata Embedding
- **Metadata Tensor**: $z$ of shape $(C, F, L)$  
  This embedding encodes external contextual information, such as weather conditions or time of day. It is added element-wise to the permuted latent representation of the main input.

---

### 3. Diffusion Embedding
- **Diffusion Timestep Embedding**: $t$  
  The timestep $t$ is encoded using a fully connected layer with SiLU activation, followed by expansion to a shape $(128, 1, 1)$. It is then processed using a **Conv1x1** layer for integration.

---

### 4. Side Information
The model incorporates:
- **Time Embedding**: Shape $(128, 1, L)$, capturing temporal dependencies.
- **Feature Embedding**: Shape $(16, F, 1)$, capturing inter-feature relationships.

These embeddings are concatenated and passed through a **Conv1x1** layer to form a block of shape $(144, F, L)$.

---

### Core Model Components

### 1. Transformer Layers
The model uses two types of transformers:
1. **Temporal Transformer**:
   - Operates on the time dimension $L$ to model temporal dependencies.
   - Input shape: $(C, F, L)$.

2. **Feature Transformer**:
   - Operates on the feature dimension $F$ to model inter-feature dependencies.
   - Input shape: $(C, F, L)$.

---

### 2. Residual Layers
Each residual layer consists of:
1. **Conv1x1 Expansion**: Expands the latent channels to $2C$.  
2. **Gated Activation Unit (GAU)**: Applies non-linear interactions in the feature space.  
3. **Conv1x1 Reduction**: Reduces the channels back to $C$.  
4. **Residual Connections**: Ensures efficient gradient flow and preserves input information.

---

### 3. Skip Connections
Skip connections aggregate outputs from multiple residual layers to:
- Improve gradient flow during training.
- Retain fine-grained information from earlier layers.

---

### 4. Output Layer
The output layer includes:
1. **Conv1x1 and ReLU**: Applies a final transformation to the processed features.
2. **Masking**: The output is multiplied by $(1 - m^{\text{co}})$, where $m^{\text{co}}$ is a conditional mask indicating observed versus missing values.

---

### Metadata Block

| **Component**           | **Description**                                                              |
|--------------------------|------------------------------------------------------------------------------|
| **Metadata Input**       | Encodes external context as an embedding block.                             |
| **Integration Method**   | Adds the metadata embedding $z$ to the permuted input latent representation. |
| **Purpose**              | Enhances model performance for tasks requiring external domain knowledge.    |
| **Examples**             | Weather conditions, time of day, economic indicators, etc.                  |

---

### Model Architecture Diagram

Below is the architecture diagram illustrating the `TimeWeaver` model:
<p align="center">
  <img alt=" csdi Time Weaver" src="https://github.com/badr07X/Synthetic-multivariate-time-series-generation-from-real-data/blob/main/figures/time%20Weaver%20.png">
</p> 



---

## Enhanced Features in TimeWeaver

1. **Metadata Embedding Block**:
   - Integrates external contextual information into the model pipeline.
   - Enables improved accuracy for tasks requiring external dependencies.

2. **Dual Transformer Layers**:
   - Separately model temporal and feature dependencies for better representation.

3. **Diffusion Framework**:
   - Uses score-based diffusion to refine predictions iteratively.

4. **Residual and Skip Connections**:
   - Ensure smooth training and efficient information propagation.

---

## Applications
TimeWeaver is designed for:
- **Time Series Imputation**:
  - Filling missing data points in multivariate time series.
- **Conditional Forecasting**:
  - Predicting future time steps based on historical data and metadata conditions.
- **Conditional Generation**:
  - Generating time series conditioned on specific input features and metadata.

---

## Advantages Over CSDI
- **Contextual Awareness**: Metadata embedding enables context-aware predictions.
- **Improved Modeling**: Dual transformers separately handle temporal and feature dependencies.
- **Enhanced Stability**: Residual and skip connections ensure smoother training.

---

## Conclusion
The TimeWeaver model extends the CSDI framework with metadata embeddings, making it well-suited for real-world time series applications where external context is critical. By leveraging transformers, residual connections, and the diffusion framework, TimeWeaver achieves state-of-the-art performance in time series imputation and generation tasks.

## Metrics 
### Joint Frechet Time Series Distance (J-FTSD)

The Joint Frechet Time Series Distance (J-FTSD) is a novel evaluation metric specifically designed for conditional time series generation. It addresses the limitations of existing metrics, such as Context-FID, which fail to penalize deviations between real and generated time series with respect to paired metadata. J-FTSD ensures that the generated data not only aligns with the statistical properties of the original time series but also adheres to the metadata conditions provided.

#### **Definition**

Given a dataset of time series and their corresponding metadata pairs $D_g = \{ (x^g, c^g) \}$, where $x^g$ denotes the time series and $c^g$ denotes the metadata, J-FTSD measures the discrepancy between the joint distributions of real and generated time series conditioned on their metadata. 

The metric projects both time series and metadata into a lower-dimensional embedding space using time series feature extractors $\phi_{\text{time}}$ and metadata feature extractors $\phi_{\text{meta}}$, respectively. The joint embeddings of the time series and metadata are then concatenated, and the Frechet Distance (FD) is computed over these embeddings.

Mathematically, J-FTSD is defined as:

$$
\text{J-FTSD}(D_g, D_r) = \| \mu_r - \mu_g \|^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{\frac{1}{2}}),
$$

where:
- $\mu_g$, $\mu_r$: Means of the joint embeddings for generated and real datasets, respectively.
- $\Sigma_g$, $\Sigma_r$: Covariance matrices of the joint embeddings for generated and real datasets, respectively.

#### **Feature Extraction**

To compute J-FTSD, the model utilizes two feature extractors:
1. **Time Series Extractor ($\phi_{\text{time}}$):** Captures the features of the time series data.
2. **Metadata Extractor ($\phi_{\text{meta}}$):** Encodes categorical and continuous metadata into a feature space.

The embeddings from these extractors are concatenated into a joint representation:

$$
z_{\text{joint}} = \text{Concat}(\phi_{\text{time}}(x), \phi_{\text{meta}}(c)).
$$

This joint representation captures the relationship between the time series and its metadata, enabling a robust comparison between real and generated distributions.

#### **Training the Feature Extractors**

The feature extractors are trained using a contrastive learning approach to ensure that the joint embeddings of time series and their corresponding metadata are mapped close together in the feature space. During training:
- Positive pairs (real time series and metadata) are pulled closer.
- Negative pairs (mismatched time series and metadata) are pushed apart.

The training loss is a combination of two objectives:
1. **Time Series Loss ($\mathcal{L}_{\text{time}}$):** Ensures accurate representation of time series features.
2. **Metadata Loss ($\mathcal{L}_{\text{meta}}$):** Optimizes the metadata embeddings.


The total loss is defined as:


Where:
- `L_time` is the loss associated with the time series features.
- `L_meta` is the loss related to the metadata features.

![Total Loss Equation](https://latex.codecogs.com/svg.latex?\mathcal{L}=\mathcal{L}_{\text{time}}%20+%20\mathcal{L}_{\text{meta}}.)



#### **Why J-FTSD?**

J-FTSD is an ideal metric for evaluating conditional time series generation models because it:
- Accurately penalizes deviations between real and generated time series while taking metadata into account.
- Captures the joint alignment of time series data and their corresponding metadata.
- Reflects the sensitivity of generated time series to metadata-specific perturbations.

#### **Applications of J-FTSD**

J-FTSD is particularly suited for use cases where the generated time series must strongly adhere to the metadata conditions, such as:
- Synthetic healthcare data generation.
- Weather forecasting conditioned on specific metadata (e.g., humidity, wind speed).
- Financial time series prediction influenced by market metadata.





