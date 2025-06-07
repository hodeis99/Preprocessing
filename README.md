## Preprocessing
This repository contains a comprehensive data preprocessing pipeline for backorder prediction datasets, preparing them for LSTM, GRU, and CNN models in supply chain management.

## Features

-  Comprehensive data cleaning (missing values, duplicates, negative values)
-  Extensive EDA (distributions, correlations, outliers)
-  Advanced class balancing (SMOTE + RandomUnderSampler)
-  Proper categorical encoding (LabelEncoder)
-  Feature scaling (MinMaxScaler)
-  Ready for deep learning models

## Dataset Overview

The dataset contains supply chain features with the target variable `went_on_backorder` indicating whether a product went on backorder.

**Original Dataset Characteristics:**
- Training samples: 1,687,861
- Test samples: 242,076
- Features: 22 (after preprocessing)
- Class imbalance: 1676567 (No) vs 11293 (Yes) in training set

**After Preprocessing:**
- Training samples: 60,000 (balanced: 40k No, 20k Yes)
- All features normalized to [0,1] range
- No missing values

## Preprocessing Steps

1. **Data Cleaning**:
   - Converted negative values to NaN
   - Dropped all rows with NaN values
   - Eliminated irrelevant columns (`sku`, `pieces_past_due`, `local_bo_qty`)
   - Removed duplicate records

2. **Exploratory Data Analysis**:
   - Statistical summaries
   - Distribution analysis (histograms, boxplots, scatter plot, pie plot)
   - Correlation matrix
   - Outlier detection and capping

3. **Feature Engineering**:
   - Categorical encoding (LabelEncoder)
   - Feature scaling (MinMaxScaler)

4. **Class Balancing**:
   - Combined SMOTE oversampling and RandomUnderSampler
   - Achieved 2:1 ratio (No:Yes)

## Getting Started

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/hodeis99/Preprocessing.git
   cd Preprocessing
