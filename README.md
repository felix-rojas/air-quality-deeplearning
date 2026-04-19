# Air-quality-deeplearning
Kaggle notebook to analyze and train a deep learning model to predict air quality (PM2.5) using:
- Latitude
- Longitude,
- Temperature,
- Hour
- Day
- Wind speed

## First (naive) approach

I decided to see if the model could figure out a relation a relationship between these data points with minimal processing.

I used all the hourly data from every available state in the USA to train the model.

The model consisted of funneling layers and a batch normalization.

The funneling was used due to its reported success on highdimensional data like images

https://bayesiandeeplearning.org/2021/papers/39.pdf

Batch normalization was added in hopes to accelerate the learning process as documented here:

https://arxiv.org/pdf/1502.03167

```python
import tensorflow as tf
from tensorflow.keras import layers, models

def build_spatial_model(input_shape):
    model = models.Sequential([

        layers.Input(shape=(input_shape,)),
        
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(), # stabilize rows
        
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        
        # Output Layer 1 neuron for PM2.5 level
        # No activation function for regression
        layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

model = build_spatial_model(len(features))
```

Despite these approaches, the results were unsatisfactory. After training for 40 epochs, the model showed a very low R-squared.

The data is scaled, so after scaling it back to real world values we obtain:
- Real-World MAE:  4.73 µg/m³
- Real-World MSE:  112.45 (µg/m³)²
- Real-World RMSE: 10.60 µg/m³
- R-squared (R2):  0.1031

### Evaluation Metrics

#### MAE

A Mean Absolute Error of 4.73 µg/m³ indicates that, on an average day, the model's prediction is off by about 4.7 µg/m³. This is a reasonable point as prediction but the larger issues stems from the extremely low R-squared.

#### MSE & RMSE

The MSE indicates a *severe* variance in the residuals. The predictions will be off due to unexpected events and this is properly reflected in the RMSE.

RMSE squares the individual errors before averaging them, which **heavily weights large mistakes**. 

RMSE (10.60) is much larger than the MAE (4.73). This proves that on "normal" days, the model might have somewhat acceptable predictions, but it fails to predict extreme pollution events (outliers).

As seen earlier, the data is prone to spikes and there are many other events which will change this:
- Wildfires
- Industrial accidents
- Firework celebrations

Some of these spikes cannot be predicted, which leads to individual prediction errors and a bad RMSE score. This is expected but somewhat large, specially compared to current methods.

#### R squared

The R2 score of 0.1031 shows that the model only explains 10.31% of the variance in the PM2.5 data. The remaining 89.69% of the fluctuations in PM2.5 levels are completely missed.

These issues probably stem from the FNN architecture. It treats every single row of data as an independent, isolated event.

### Steps forward

The R2 score is quite telling that our model configuration is not adequate for these predictions. 

After a literature review, there have been reported shortcomings in this funnel architecture and my techniques for spatial-temporal datapoints were lacking.

#### Next goals and approach

I will improve the spatial-temporal interpretation of the data.
In preprocessing, I need to asociate the time and space sequences.

For this next approach I will use a LSTM configuration, so that the model learns the associations between time and space and PM2.5 levels.

## Second approach

### Preprocessing

#### Cyclical encoding

This is necessary for the model to recognize that the data will fluctuate naturally, rather than a linear representation like before. The linear representation introduces artificial distances in data that should be considered contiguous:

> December is right next to January but my previous interpretation gave it a distance of 11 since 12-1 = 11. 

Using sine and cosine functions we can embed periodicity to this data into circles:

1. The hours in a day cycle

```python
hour_sin = np.sin(2 * np.pi * hour_of_day / HOURS_IN_DAY)
hour_cos = np.cos(2 * np.pi * hour_of_day / HOURS_IN_DAY)
```

2. The days in a year cycle

```python
day_sin = np.sin(2 * np.pi * day_of_year / DAYS_IN_YEAR)
day_cos = np.cos(2 * np.pi * day_of_year / DAYS_IN_YEAR)
```

#### Time-space grouping

Data is now clustered around location, using sorted time values. I'm naively using a 7 day grouping of data so that the model can infer a relation between the series, using the lat and lon pairs as ID's for locations.
