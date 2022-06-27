#######################################################################################################################
# CS5014 Machine Learning: Practical 1 - Methodology (Credit Approval)
#
# Main Python script for the practical.
#
# Author: 170004680
#######################################################################################################################

# Imports:

import pandas as pd
from P1functions import createDummies
from P1functions import processData
from P1functions import evaluateDataSet
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from P1functions import processDataAdv  # For advanced tasks.
from sklearn.preprocessing import PolynomialFeatures  # For advanced tasks.

# Constants:

DATA_FILE = "../data/crx.data"  # Where to get the input data from.
OUTPUT_HEADER = "A16"  # Column which is the output of the logistic regression.

COLUMNS_TO_DUMMY = ['A1', 'A4', 'A5', 'A6', 'A7', 'A9', 'A10', 'A12', 'A13']  # Column names that require dummies.

TRAINING_SET_PROPORTION = 0.8  # Use 80% of the input data for training, 20% for testing.
RANDOM_SAMPLING_SEED = 1010  # Seed ensures (same) random sampling when splitting for reproducibility.

INPUT_OUTPUT_CORR_MIN = 0.3  # Threshold by which features have to be correlated with the output to not be redundant.
INPUT_INPUT_CORR_MAX = 0.9  # Maximum correlation between features to enforce linear independence.


#######################################################################################################################
# Part 1: Data Processing.
#######################################################################################################################

# 1.1. Load data from CSV file, parse missing values to NaN (null), and assign labels to the columns.
crxDataFrame = pd.read_csv(DATA_FILE, ",", na_values=['?'], names=['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9',
                                                                   'A10', 'A11', 'A12', 'A13', 'A14', 'A15', OUTPUT_HEADER])

# 1.2. Drop all rows that have null (NaN) values.
# There are no infinities or invalid categorical values in any column (feature) to remove.
crxDataFrame = crxDataFrame.dropna()

# 1.3. Drop any duplicate rows as we don't want equivalent records to end up in both the training and testing set.
crxDataFrame = crxDataFrame.drop_duplicates()

# 1.4. Create dummy variables:
# Columns A1, A4, A5, A6, A7, A9, A10, A12, and A13 require dummy variables for their non-continuous values.
# A16 is the output so it requires no dummy despite having non-continuous values.
crxDataFrame = createDummies(crxDataFrame, COLUMNS_TO_DUMMY)

# 1.5. Encode A16 column, where: '-' = 0, '+' = 1.
crxDataFrame[OUTPUT_HEADER] = crxDataFrame[OUTPUT_HEADER].replace(['+', '-'], [1, 0])

# 1.6. Split data into training and testing sets (80% training set, 20% testing set).
# Data seems to be sorted. There are oscillating blocks of +/- classifications, so sample at random when splitting.
# Split data in stratified fashion (preserve the proportion of output classes in the data).
# This maintains the distribution in the sets, so the training/testing sets should be ~44.5% '+', ~55.5% '-'.
crxDataFrameOutput = crxDataFrame[OUTPUT_HEADER]
crxDataFrameInput = crxDataFrame.drop(OUTPUT_HEADER, axis=1)

crxOriginalTrainingInput, crxOriginalTestingInput, crxOriginalTrainingOutput, crxOriginalTestingOutput = \
    train_test_split(crxDataFrameInput, crxDataFrameOutput, train_size=TRAINING_SET_PROPORTION,
                     random_state=RANDOM_SAMPLING_SEED, stratify=crxDataFrameOutput)

# 1.7. Pre-process the data: feature selection and scaling.
crxTrainingSetInput, crxTrainingSetOutput, selectedFeatureNames, scaler = \
    processData(crxOriginalTrainingInput, crxOriginalTrainingOutput, INPUT_OUTPUT_CORR_MIN, INPUT_INPUT_CORR_MAX,
                None, None)

# Data Leakage:
#   Prevented as no data dependant actions are completed before the data set is split into a training and testing set.
#   Also, duplicate rows are removed, if they exist, before splitting so the same data could not accidentally end up in
#   both the training and testing sets. Thus, none of the data in the training set has 'knowledge' of the data in the
#   test set.


#######################################################################################################################
# Part 2: Training.
#######################################################################################################################

print("Part 2: Training\n")

# 2.1 Train using no penalty and no class weights logistic regression on the standardised training input.
logReg_NoPenNoClassWeight = LogisticRegression(penalty='none', class_weight='none')
logReg_NoPenNoClassWeight.fit(crxTrainingSetInput, crxTrainingSetOutput)

# Show training results for Logistic Regression with penalty='none', class_weight='none'.
# Note: Not strictly needed, we care about performance with the testing set.
evaluateDataSet(logReg_NoPenNoClassWeight, crxTrainingSetInput, crxTrainingSetOutput, 'Training', 'none', 'none')

# 2.2 Train using no penalty and balanced class weights logistic regression on the standardised training input.
logReg_NoPenBalClassWeight = LogisticRegression(penalty='none', class_weight='balanced')
logReg_NoPenBalClassWeight.fit(crxTrainingSetInput, crxTrainingSetOutput)

# Show training results for Logistic Regression with penalty='none', class_weight='balanced'.
# Note: Not strictly needed, we care about performance with the testing set.
evaluateDataSet(logReg_NoPenBalClassWeight, crxTrainingSetInput, crxTrainingSetOutput, 'Training', 'none', 'balanced')


#######################################################################################################################
# Part 3: Evaluation.
#######################################################################################################################

print("Part 3: Evaluation\n")

# 3.1. Get the input and output testing sets for evaluation by pre-processing the data the same as the training set:
# The same encoding, selected features, and scaler as the training set are used.
crxTestingSetInput, crxTestingSetOutput, selectedFeatureNames, scaler = \
    processData(crxOriginalTestingInput, crxOriginalTestingOutput, INPUT_OUTPUT_CORR_MIN, INPUT_INPUT_CORR_MAX,
                selectedFeatureNames, scaler)

# 3.2. Evaluate the testing set using Logistic Regression with penalty='none', class_weight='none'.
evaluateDataSet(logReg_NoPenNoClassWeight, crxTestingSetInput, crxTestingSetOutput, 'Testing', 'none', 'none')

# 3.3. Evaluate the testing set using Logistic Regression with penalty='none', class_weight='balanced'.
evaluateDataSet(logReg_NoPenBalClassWeight, crxTestingSetInput, crxTestingSetOutput, 'Testing', 'none', 'balanced')


#######################################################################################################################
# Part 4: Advanced Tasks.
#######################################################################################################################

print("Part 4: Advanced Tasks\n")

# 4.a. Set the penalty parameter to 'l2' for a LogisticRegression classifier.
logReg_NoPenNoClassWeight = LogisticRegression(penalty='none', class_weight='none', max_iter=1000)  # No regularisation.
logReg_L2PenNoClassWeight = LogisticRegression(penalty='l2', class_weight='none', max_iter=1000)  # L2 Regularisation.

# 4.b. Implement a 2nd degree polynomial expansion on the training data set.
# This does not include the interactions of features with themselves.
polynomialExpander = PolynomialFeatures(degree=2, interaction_only=True)

# Apply the polynomial expansion to the training and testing data sets.
crxL2TrainingSetInputExpanded = pd.DataFrame(polynomialExpander.fit_transform(crxOriginalTrainingInput))
crxL2TestingSetInputExpanded = pd.DataFrame(polynomialExpander.transform(crxOriginalTestingInput))

# Show results of the expansion.
print("No: of Features Before 2nd Degree Polynomial Expansion: " + str(len(crxOriginalTrainingInput.columns)))
print("No: of Features After 2nd Degree Polynomial Expansion: " + str(len(crxL2TrainingSetInputExpanded.columns)) + "\n")

# Process the training and testing data the same as for previous parts, but do not use feature selection (L2 penalty will do it).
crxL2TrainingSetInputExpanded, crxL2TrainingSetOutput, l2Scaler = processDataAdv(crxL2TrainingSetInputExpanded, crxOriginalTrainingOutput, None)
crxL2TestingSetInputExpanded, crxL2TestingSetOutput, l2Scaler = processDataAdv(crxL2TestingSetInputExpanded, crxOriginalTestingOutput, l2Scaler)

# 4.c. Compare the use of regularised and un-regularised classifiers on the expanded data set.
logReg_NoPenNoClassWeight.fit(crxL2TrainingSetInputExpanded, crxL2TrainingSetOutput)  # No regularisation training.
logReg_L2PenNoClassWeight.fit(crxL2TrainingSetInputExpanded, crxL2TrainingSetOutput)  # L2 regularisation training.

# Evaluate the classifier with no penalty on the testing set.
evaluateDataSet(logReg_NoPenNoClassWeight, crxL2TestingSetInputExpanded, crxL2TestingSetOutput, 'Testing (Expanded)', 'none', 'none')
# Evaluate the classifier with L2 penalty on the testing set.
evaluateDataSet(logReg_L2PenNoClassWeight, crxL2TestingSetInputExpanded, crxL2TestingSetOutput, 'Testing (Expanded)', 'L2', 'none')


#######################################################################################################################
# End.
#######################################################################################################################
