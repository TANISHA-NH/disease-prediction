Here's a GitHub-friendly README template you can use directly:

```markdown
# Disease Prediction Project

This project aims to predict various diseases using machine learning models. The repository contains different scripts, models, and datasets for disease prediction, including anemia, diabetes, heart disease, and genetic conditions.

## Table of Contents
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Usage](#usage)
- [Models](#models)
- [Scripts Overview](#scripts-overview)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/disease-prediction.git
   cd disease-prediction
   ```

2. **Install required dependencies:**
   Make sure you have Python 3.8+ installed. Then, install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   If a `requirements.txt` file is not available, the following packages are commonly used in such projects:
   ```bash
   pip install numpy pandas scikit-learn tensorflow keras joblib
   ```

3. **Setup Models and Data Files:**
   Ensure that the `.h5` and `.pkl` model files are available in the same directory. If required datasets are not present, download and place them in the appropriate folders.

## Project Structure

The project is organized as follows:

```
.
├── Anemiaimages/          # Contains images for anemia prediction
├── bloodtestmodels/       # Machine learning models for blood test data
├── btmtraining/           # Training scripts or datasets for blood test models
├── disease-prediction/    # Main folder for disease prediction models
├── nandhimages/           # Additional images or datasets
├── an1_predictor.py       # Script for anemia prediction
├── Anemia.joblib          # Pre-trained model for anemia prediction
├── cnn_model.h5           # CNN model for image-based predictions
├── di1_predictor.py       # Script for diabetes prediction
├── Diabetes.joblib        # Pre-trained model for diabetes
├── genome(test+predict).py # Script for genetic prediction
├── GenomePredictor.pkl    # Pre-trained model for genome-based predictions
├── h1_predictor.py        # Script for heart disease prediction
├── heartimg.h5            # Pre-trained model for heart disease image-based predictions
├── main.py                # Main entry script for running different predictions
├── output.csv             # File for storing prediction outputs
├── PheGen1.csv            # Sample genome data
├── sickleimg.py           # Script for sickle cell disease prediction
└── requirements.txt       # List of required dependencies
```

## Datasets

- `Anemiaimages/`, `nandhimages/`, and other folders may contain training and testing datasets for different conditions.
- `.csv` files like `PheGen1.csv` contain data for genome-based predictions.

## Usage

1. **Running the Main Script:**
   The `main.py` script serves as the central entry point for predictions.
   ```bash
   python main.py
   ```
   Modify the script as needed to specify which model to run.

2. **Running Individual Predictors:**
   Each condition can be predicted using its respective script:
   - **Anemia Prediction:**
     ```bash
     python an1_predictor.py
     ```
   - **Diabetes Prediction:**
     ```bash
     python di1_predictor.py
     ```
   - **Heart Disease Prediction:**
     ```bash
     python h1_predictor.py
     ```
   - **Genomic Predictions:**
     ```bash
     python genome(test+predict).py
     ```

3. **Model Training (if needed):**
   Use scripts in the `btmtraining/` directory or add your training scripts to retrain models using custom datasets.

## Models

- **Pre-trained Models:** The models are stored in files such as `.h5`, `.joblib`, and `.pkl` for TensorFlow/Keras, scikit-learn, and Pickle formats respectively.
- **Model Details:**
  - `cnn_model.h5`: A convolutional neural network for image-based predictions.
  - `Anemia.joblib`, `Diabetes.joblib`: Models trained using blood test data.
  - `GenomePredictor.pkl`: A genetic prediction model.

## Scripts Overview

- **an1_predictor.py:** Loads the anemia model and makes predictions based on input data.
- **di1_predictor.py:** Predicts diabetes using the corresponding model.
- **h1_predictor.py:** Uses a pre-trained model to predict heart disease.
- **genome(test+predict).py:** Script for predicting diseases based on genome data.
- **sickleimg.py:** Prediction for sickle cell disease using an image-based model.

## Results

The prediction results can be output to `output.csv` or displayed in the console. Each script may need to be adjusted to specify the output format.

## Contributing

Contributions are welcome. Please create a new branch for each feature or bug fix, and submit a pull request for review.

