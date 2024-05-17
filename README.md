# AI-Assignment-2-Predicting-diabetes-with-ANN
Predicting diabetes with ANN

Overview

This project aims to predict diabetes status using an Artificial Neural Network (ANN). The dataset includes various health indicators and demographic information. The goal is to classify individuals into three categories: Non-Diabetic, Pre-Diabetic, and Diabetic.
Dataset

The dataset used in this project contains the following columns:
‎

    • No_Pation: Patient number (unique identifier, excluded from analysis)
    • Gender: Gender of the patient
    • Age: Age of the patient
    • CLASS: Diabetes status (Non-Diabetic, Pre-Diabetic, Diabetic)
    • Various health-related numeric features

Steps Performed
 
    1. Load and Preprocess Data:
        • Load the dataset from a CSV file.
        • Handle missing values by dropping rows with any missing entries.
        • Strip leading/trailing spaces from the 'CLASS' column and map class labels to numeric values.
        • Clean and standardise the 'Gender' column.
 
    2. Exploratory Data Analysis (EDA):
        • Visualise the distribution of the 'Gender' column.
        • Visualise the distribution of each numeric feature.
        • Visualise the relationship between numeric features and the 'CLASS' column using boxplots.
        • Encode categorical columns to numeric for correlation analysis.
        • Create a pairplot of numeric features.
        • Visualise the correlation matrix.
 
    3. Model Building and Evaluation:
        • Encode the 'Gender' column using label encoding.
        • Define feature variables (X) and target variable (y).
        • Scale the features using StandardScaler.
        • Split the data into training and testing sets.
        • Build an ANN model with L2 regularization.
        • Compile and train the model using early stopping to prevent overfitting.
        • Evaluate the model using confusion matrix, classification report, accuracy, and precision scores.
        • Visualise the confusion matrix and training history.

Requirements

    • Python 3.x
    • Pandas
    • NumPy
    • Matplotlib
    • Seaborn
    • Scikit-learn
    • TensorFlow

How to Run
 
    1. Clone the Repository:

        git clone <repository_url>
        cd <repository_directory>
 
    2. Install Dependencies:

        pip install -r requirements.txt
 
    3. Run the Notebook:
        Open and run the provided Jupyter notebook to execute all steps.

Results
 
    • Confusion Matrix: Provides insight into the performance of the model by showing the counts of true positive, true negative, false positive, and false negative predictions.
    • Accuracy: Overall accuracy of the model on the test set.
    • Precision: Precision scores (macro, micro, and weighted) to evaluate the quality of the predictions.
    • Training History: Visualises the training and validation accuracy and loss over epochs.

Conclusion

This project demonstrates the process of predicting diabetes status using an ANN. The steps include data preprocessing, EDA, model building, evaluation, and visualization of results. The ANN model,            trained with appropriate regularization and early stopping, provides valuable insights into diabetes prediction based on the given dataset.

Author

Vansh Karia

License

This project does not have a license.
