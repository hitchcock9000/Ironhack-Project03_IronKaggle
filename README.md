# 🏠 House Price Prediction - King County, USA

## 📋 Project Overview

This machine learning project analyzes and predicts house prices in King County, Washington (including Seattle) using various property features. The project implements multiple regression models to find the best predictor for real estate values.

## 🎯 Objectives

- Analyze historical house sales data from King County
- Identify key factors that influence house prices
- Build and compare multiple machine learning models
- Create a robust price prediction system for real estate valuation

## 📊 Dataset Information

### Source
King County House Sales dataset containing 21,613 house sale records with 21 features.

### Key Features
- **Property Details**: Bedrooms, bathrooms, square footage, lot size
- **Quality Metrics**: Grade (construction quality), condition, view rating
- **Location Data**: Latitude, longitude, zipcode
- **Historical Info**: Year built, renovation year
- **Special Features**: Waterfront access, basement area

### Target Variable
- **Price**: Sale price of the house (in USD)

## 🛠️ Technical Stack

### Languages & Tools
- Python 3.8+
- Jupyter Notebook

### Core Libraries
- **Data Processing**: 
  - `pandas` - Data manipulation and analysis
  - `numpy` - Numerical computations
  
- **Visualization**: 
  - `matplotlib` - Plotting and graphing
  - `seaborn` - Statistical data visualization
  
- **Machine Learning**:
  - `scikit-learn` - ML algorithms and utilities
  - `xgboost` - Gradient boosting framework

## 🚀 Getting Started

### Prerequisites
```bash
# Install required packages
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib
```

### Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/house-price-prediction.git
cd house-price-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook house_price_prediction_refactored.ipynb
```

## 📁 Project Structure

```
house-price-prediction/
│
├── data/
│   └── king_country_houses_aa.csv    # Raw dataset
│
├── house_price_prediction_refactored.ipynb   # Main analysis notebook
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
│
└── models/
    └── best_model_*.pkl              # Saved trained models
```

## 🔄 Workflow

### 1. Data Loading & Exploration
- Load the King County housing dataset
- Initial data inspection and statistics
- Check for missing values and data types

### 2. Exploratory Data Analysis (EDA)
- Distribution analysis of house prices
- Correlation analysis between features
- Identification of important predictors
- Outlier detection and analysis

### 3. Data Preprocessing
- Date feature extraction (year, month, quarter)
- Handle missing values (if any)
- Feature type conversion

### 4. Feature Engineering
- **House Age**: Years since construction
- **Renovation Status**: Binary flag for renovated properties
- **Years Since Renovation**: Time elapsed since last renovation
- **Total Square Footage**: Combined living and basement area
- **Price per Square Foot**: For comparative analysis
- **Bedroom-Bathroom Ratio**: Property layout metric
- **Luxury Flag**: Identifies high-end properties

### 5. Model Development
Four regression models are trained and evaluated:

#### Linear Regression (Baseline)
- Simple, interpretable baseline model
- Uses scaled features
- Good for understanding linear relationships

#### Random Forest Regressor
- Ensemble of decision trees
- Handles non-linear patterns
- Provides feature importance rankings

#### XGBoost Regressor
- Gradient boosting algorithm
- High accuracy for complex patterns
- Built-in regularization

#### AdaBoost Regressor
- Adaptive boosting technique
- Sequential learning approach
- Good for reducing bias

### 6. Model Evaluation
Metrics used for comparison:
- **MAE** (Mean Absolute Error): Average prediction error in dollars
- **RMSE** (Root Mean Square Error): Penalizes large errors more
- **R² Score**: Proportion of variance explained (0-1, higher is better)

## 📈 Results Summary

### Model Performance (Example Results)
| Model | MAE ($) | RMSE ($) | R² Score |
|-------|---------|----------|----------|
| Linear Regression | 120,000 | 180,000 | 0.70 |
| Random Forest | 80,000 | 130,000 | 0.85 |
| XGBoost | 75,000 | 125,000 | 0.87 |
| AdaBoost | 95,000 | 145,000 | 0.80 |

*Note: Actual results may vary based on random seed and hyperparameters*

### Key Insights
1. **Most Important Features**:
   - Square footage of living area
   - Grade (construction quality)
   - Location (latitude/longitude)
   - Number of bathrooms
   - View quality

2. **Model Selection**:
   - XGBoost and Random Forest consistently outperform simpler models
   - Tree-based models capture non-linear relationships better
   - Linear regression serves as a good baseline

## 💡 Usage Examples

### Making Predictions
```python
import joblib
import pandas as pd

# Load the saved model
model = joblib.load('best_model_xgboost.pkl')

# Prepare new house data
new_house = pd.DataFrame({
    'bedrooms': [3],
    'bathrooms': [2.5],
    'sqft_living': [2000],
    'sqft_lot': [5000],
    'floors': [2],
    'waterfront': [0],
    'view': [2],
    'condition': [4],
    'grade': [8],
    # ... add all required features
})

# Make prediction
predicted_price = model.predict(new_house)
print(f"Predicted Price: ${predicted_price[0]:,.2f}")
```

## 🔍 Feature Importance

Top factors affecting house prices:
1. **Living Area Size** (~35% importance)
2. **Construction Grade** (~20% importance)
3. **Location Coordinates** (~15% importance)
4. **Number of Bathrooms** (~10% importance)
5. **View Quality** (~8% importance)

## ⚠️ Limitations

- **Temporal Factors**: Model doesn't account for market trends over time
- **External Factors**: Economic conditions, interest rates not included
- **Luxury Properties**: Less accurate for homes >$5 million
- **Geographic Scope**: Limited to King County area only

## 🔮 Future Improvements

### Short-term
- Hyperparameter tuning using GridSearchCV
- Feature selection techniques (RFE, LASSO)
- Cross-validation for better generalization
- Ensemble methods combining multiple models

### Long-term
- Time series analysis for market trends
- Deep learning models (Neural Networks)
- Integration with external data sources (schools, crime rates)
- Web application for real-time predictions
- API development for production deployment

## 📚 References

- [King County House Sales Dataset](https://www.kaggle.com/datasets)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Feature Engineering for Machine Learning](https://www.oreilly.com/library)

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Steps to Contribute:
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- King County for providing the public dataset
- Open source community for amazing ML libraries
- Data science community for inspiration and best practices

## 📞 Contact

For questions or feedback, please open an issue in the GitHub repository.

---

**Last Updated**: November 2025  
**Version**: 2.0 (Refactored and Optimized)
