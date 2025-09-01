# YouTube Influencer Analysis Project

A comprehensive data science project that analyzes YouTube influencer data to predict video performance and understand factors that contribute to high viewership.

## 📊 Project Overview

This project analyzes YouTube influencer data to:
- Perform exploratory data analysis on YouTube video metrics
- Build machine learning models to predict high-performing videos (Bucket 4 & 5)
- Identify key factors that contribute to video success
- Provide insights for content creators and marketers

## 🎯 Key Features

- **Data Analysis**: Comprehensive EDA with custom YouTube-themed visualizations
- **Machine Learning**: Binary classification models to predict high-performing videos
- **Model Evaluation**: Extensive model performance analysis with calibration techniques
- **Custom Utilities**: Reusable functions for plotting and model evaluation
- **Results Analysis**: Detailed analysis of prediction results and insights

## 📁 Project Structure

```
youtube-analysis/
├── EDA.ipynb              # Exploratory Data Analysis
├── model.ipynb            # Machine Learning Model Development
├── result.ipynb           # Results Analysis and Insights
├── utils.py               # Utility functions for plotting and evaluation
└── README.md              # Project documentation
```

## 🚀 Getting Started

### Prerequisites

Install the required Python packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost kagglehub chardet
```

### Data Setup

The project uses the YouTube Influencers Dataset from Kaggle. The data is automatically downloaded using `kagglehub` in the EDA notebook.

### Running the Analysis

1. **Start with EDA**: Open `EDA.ipynb` to explore the dataset and understand the data structure
2. **Build Models**: Run `model.ipynb` to develop and evaluate machine learning models
3. **Analyze Results**: Use `result.ipynb` to examine prediction results and insights

## 📈 Analysis Components

### 1. Exploratory Data Analysis (`EDA.ipynb`)
- Data loading and preprocessing
- Custom YouTube-themed visualization setup
- Comprehensive data exploration
- Feature engineering and data cleaning

### 2. Machine Learning Models (`model.ipynb`)
- Data preparation and feature scaling
- Multiple classification algorithms:
  - Logistic Regression
  - Support Vector Machine (SVM)
  - XGBoost
- Model evaluation with cross-validation
- Calibration techniques (Platt Scaling, Isotonic Regression)
- Feature importance analysis

### 3. Results Analysis (`result.ipynb`)
- Prediction performance analysis
- Bucket analysis for high-performing videos
- Statistical summaries of predicted results
- Insights for content creators

### 4. Utility Functions (`utils.py`)
- `plot_bucket_trends()`: Visualize trends across video performance buckets
- `model_evaluation()`: Comprehensive model evaluation with multiple metrics
- `compare_calibration_curves()`: Compare different calibration methods

## 🎨 Custom Visualizations

The project includes a custom YouTube-themed color palette and styling:
- **Colors**: Red (#FF0000), Yellow (#FBD704), Blue (#10D8B8), Purple (#261D44)
- **Custom Font**: Hanken Grotesk for professional presentation
- **Consistent Theme**: Clean, modern visualizations optimized for data storytelling

## 📊 Key Metrics Analyzed

- **Video Views**: Primary target variable
- **Channel Metrics**: Subscribers, total views, video count
- **Content Features**: Duration, video quality, playlist count
- **Performance Scores**: Loyalty, attractiveness, and efficiency scores
- **Creator Demographics**: Gender and other creator information

## 🎯 Model Performance

The project focuses on predicting high-performing videos (Bucket 4 & 5) which typically have:
- **Minimum Views**: 1,354,030
- **Maximum Views**: 198,286,476
- **Mean Views**: 9,317,603
- **Median Views**: 8,060,281

## 🔧 Technical Details

- **Language**: Python 3.x
- **Key Libraries**: pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost
- **Data Source**: Kaggle YouTube Influencers Dataset
- **Model Types**: Binary classification with probability calibration
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC, Brier Score

## 📝 Usage Notes

1. **Font Setup**: The project uses a custom font (Hanken Grotesk). If you don't have this font installed, the visualizations will fall back to system defaults.

2. **Data Path**: Update the data path in the notebooks if you're using a different dataset location.

3. **Kaggle API**: Ensure you have Kaggle API credentials set up for automatic dataset download.

## 🤝 Contributing

Feel free to contribute to this project by:
- Adding new analysis methods
- Improving visualizations
- Enhancing model performance
- Adding new features or insights

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Dataset provided by Kaggle community
- YouTube for the platform that enables this analysis
- Open source community for the tools and libraries used

---

**Note**: This project is for educational and research purposes. Always respect YouTube's terms of service and data usage policies when working with YouTube data.
