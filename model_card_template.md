# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
– Person or organization developing model: Borja Sánchez
– Model date: 2023-09-17
– Model version: 1
– Model type: Logistic Regression
– Information about training algorithms, parameters, fairness constraints or other applied approaches, and features
– Paper or other resource for more information
– Citation details: https://github.com/borja-sc/mlops-census-bureau-classification
– License: see https://github.com/borja-sc/mlops-census-bureau-classification
– Where to send questions or comments about the model: https://github.com/borja-sc

## Intended Use
– Primary intended uses: The model is intended to serve as toy model for a project to predict income level from census data.
– Primary intended users: Students of MLOps
– Out-of-scope use cases: Real-world applications

## Training Data
The dataset https://archive.ics.uci.edu/dataset/20/census+income is used for the project.

A preprocessing where rows with NA are dropped is applied.
Categorical columns are one-hot-encoded and continuous variables are scaled.

An 80/20 train/test split is applied to the dataset.

## Evaluation Data
The dataset https://archive.ics.uci.edu/dataset/20/census+income is used for the project.

A preprocessing where rows with NA are dropped is applied.
Categorical columns are one-hot-encoded and continuous variables are scaled.

An 80/20 train/test split is applied to the dataset.

## Metrics
- F1 score: 0.6643382352941176
- Accuracy: 0.8486468564797547

## Ethical Considerations
No ethical considerations are devised for this model.

## Caveats and Recommendations
This model has plenty of margin for improvement and should not be used for serious applications.