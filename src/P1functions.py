#######################################################################################################################
# CS5014 Machine Learning: Practical 1 - Methodology (Credit Approval)
#
# Auxiliary functions required for the P1main script.
#
# Author: 170004680
#######################################################################################################################

# Imports:

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from matplotlib import pyplot
from sklearn.metrics import auc  # Used to confirm AP is the area under the precision-recall curve.
from scipy import stats  # For outlier checking.
import numpy as np  # For outlier checking.


#######################################################################################################################
# Functions:
#######################################################################################################################

# Process the data by performing feature selection and scaling.
#
# Params: crxInputSet - The input data set to be processed.
#         crxOutputSet - The output data set to be processed.
#         inputOutputCorrMin - Threshold for how correlated features have to be with the output to be retained.
#         inputInputCorrMax - Threshold for how correlated features have to be with each other to be removed.
#         selectedFeatureNames - List of column headings for selected features, 'None' if no features selected yet.
#         scaler - StandardScaler to use for standardisation. 'None' when processing training, given for testing.
#
# Return: The processed input set, the processed output set, the set of features selected, and the scaler used.
def processData(crxInputSet, crxOutputSet, inputOutputCorrMin, inputInputCorrMax, selectedFeatureNames, scaler):
    if selectedFeatureNames is None:  # I.e., the training set.

        # Feature Selection: Reduce the features in the input training data set using Pearson Correlation.
        crxInputSet_FeaturesReduced = featureSelectionCorr(crxInputSet, crxOutputSet, crxOutputSet.name,
                                                           inputOutputCorrMin, inputInputCorrMax)
        # Retain features used in model for evaluating with the testing set.
        selectedFeatureNames = list(crxInputSet_FeaturesReduced.columns)

    else:  # I.e., the testing set.

        # Get the features required by the trained model from the testing set.
        crxInputSet_FeaturesReduced = crxInputSet[crxInputSet.columns.intersection(selectedFeatureNames)]

    # Data Standardisation:
    # crxInputSet_FeaturesReduced.describe() shows the distribution of the data - needs to be scaled.
    # zScores = np.abs(stats.zscore(crxInputSet_FeaturesReduced['A8']))
    # print(np.where(zScores > 3))  # Revealed existence of outliers, will standardise the data.
    if scaler is None:  # I.e., the training set.
        # If training set, then fit.
        scaler = StandardScaler().fit(crxInputSet_FeaturesReduced)

    # Use the scaler to transform the data. For the testing set, this will be the same scaler as used for training.
    crxInputSet_Standardised = scaler.transform(crxInputSet_FeaturesReduced)

    # Convert to DataFrame for ease.
    crxInputSet_Standardised = pd.DataFrame(crxInputSet_Standardised, columns=crxInputSet_FeaturesReduced.columns)

    return crxInputSet_Standardised, crxOutputSet, selectedFeatureNames, scaler


# Method for creating dummy variables for attributes in a DataFrame. This method adds dummies to the frame and removes
# the old attribute the dummy variables result from.
#
# Params: dataFrame - Data frame containing the attributes to create dummy variables for and add said dummies to.
#         columnsToDummy - List of attribute names in the data frame to create dummy variables for.
#
# Returns: Data frame containing all dummy variable columns and none of the columns the dummy variables came from.
def createDummies(dataFrame, columnsToDummy):
    for columnToDummy in columnsToDummy:
        # Concatenate the data frame with the dummies created for the current attribute.
        # Assign original column prefix as many attributes have the same non-continuous value domain.
        dataFrame = pd.concat([dataFrame, pd.get_dummies(dataFrame[columnToDummy], prefix=columnToDummy)], axis=1)
        # Remove column the dummy variables result from.
        dataFrame.drop(columnToDummy, axis=1, inplace=True)

    return dataFrame


# Feature reduction via filtering. Note, better accuracy can possibly be achieved using embedded feature selection.
# Features are selected using Pearson correlation filtering, which assesses features based on correlation.
# Features have to be 'strongly' correlated with the output variable and 'weakly' correlated with each other.
#
# Params: inputDataFrame - The input features before feature selection is applied.
#         outputDataFrame - The output series.
#         outputHeader - Name of the output series (column header).
#         outputCorrelationThreshold - Threshold for how correlated features have to be with the output to be retained.
#         inputsCorrelationThreshold - Threshold for how correlated features have to be with each other to be removed.
#
# Return: New input data frame with the selected features only.
def featureSelectionCorr(inputDataFrame, outputDataFrame, outputHeader, outputCorrelationThreshold,
                         inputsCorrelationThreshold):
    # Check correlation between each feature and the output. Remove features with weak correlation to the output.
    correlations, dataFrame = featureSelectionInputOutputCorr(inputDataFrame, outputDataFrame,
                                                              outputCorrelationThreshold)

    # Check correlation between pairs of features. Remove strongly correlated features for linear independence.
    dataFrame = featureSelectionInputInputCorr(dataFrame, correlations, outputHeader, inputsCorrelationThreshold)

    return dataFrame


# Reduce features in a given input data frame based on their correlation with the output. Features that are weakly
# correlated with the output are removed.
#
# Params: inputDataFrame - The input features before feature selection is applied.
#         outputDataFrame - The output series.
#         outputCorrelationThreshold - Threshold for how correlated features have to be with the output to be retained.
#
# Return: New input data frame with the selected features only based on correlations with the output.
def featureSelectionInputOutputCorr(inputDataFrame, outputDataFrame, outputCorrelationThreshold):
    # Calculate the correlation plot of the data set (features and output column).
    dataFrame = pd.concat([inputDataFrame, outputDataFrame], axis=1)
    correlations = dataFrame.corr()

    # Get correlations with the output variable.
    correlationTarget = abs(correlations[outputDataFrame.name])
    # Select highly correlated features (features that meet the threshold for correlation with the output).
    selectedFeatures = correlationTarget[correlationTarget >= outputCorrelationThreshold]
    # Get a DataFrame of just the selected features.
    dataFrame = dataFrame.loc[:, selectedFeatures.index.values.tolist()]
    dataFrame.drop(outputDataFrame.name, axis=1, inplace=True)  # Don't include the output column in selected features.

    return correlations, dataFrame


# Reduce features in a given input data frame based on their correlation with each other. Features that are strongly
# correlated with each other are reduced. In a feature pair which is strongly correlated, the one that is less correlated
# with the output is removed.
#
# Params: dataFrame - The input features and output series (here, the input series has already been somewhat reduced).
#         correlations - Correlation data between all of the original features and the output.
#         outputHeader - Name of the output series (column header).
#         inputsCorrelationThreshold - Threshold for how correlated features have to be with each other to be removed.
#
# Return: New input data frame with the selected features based on correlations between pairs of figures.
def featureSelectionInputInputCorr(dataFrame, correlations, outputHeader, inputsCorrelationThreshold):
    # Check the correlations between features.
    # If highly correlated, then remove as they are redundant (linearly dependant) features.
    selectedFeatureCorrelations = abs(dataFrame.corr())
    for i in range(selectedFeatureCorrelations.shape[0]):
        for j in range(i + 1, selectedFeatureCorrelations.shape[0]):

            # For each pair of selected features, if they are 'highly' correlated with each other, then remove the
            # feature that has the weakest correlation with the output.
            if selectedFeatureCorrelations.iloc[i, j] >= inputsCorrelationThreshold:

                featureHeaderI = selectedFeatureCorrelations.columns[i]
                correlationIOutput = correlations[featureHeaderI][outputHeader]
                featureHeaderJ = selectedFeatureCorrelations.columns[j]
                correlationJOutput = correlations[featureHeaderJ][outputHeader]

                # Drop the lesser correlated feature with the output.
                if correlationIOutput > correlationJOutput:
                    if featureHeaderJ in dataFrame.columns:
                        dataFrame.drop(featureHeaderJ, axis=1, inplace=True)  # Drop column j from dataFrame.
                else:
                    if featureHeaderI in dataFrame.columns:
                        dataFrame.drop(featureHeaderI, axis=1, inplace=True)  # Drop column i from dataFrame.
                    break

    return dataFrame


# Evaluate a data set by using the provided input features and logistic regression model against the expected outputs.
#
# Params: logReg - Logistic Regression model, described by penalty and class_weight parameters.
#         inputFeatures - Input for the model.
#         expectedOutput - The expected output for the data set given the inputs.
#         dataSetType - Type of data set being evaluates such as 'training' or 'testing'
#         penalty - Penalty parameter used in the logReg model.
#         class_weight - Class weight parameters used in the logReg model.
def evaluateDataSet(logReg, inputFeatures, expectedOutput, dataSetType, penalty, class_weight):
    # Predict the outcome of the input set features using the provided logistic regression model.
    crxTestingSetOutputHat = logReg.predict(inputFeatures)

    # Get the predicted probabilities (the probability for each class).
    crxTestingSetOutputPredictProba = logReg.predict_proba(inputFeatures)
    crxTestingSetOutputPredictProba = crxTestingSetOutputPredictProba[:, 1]  # Positive class probabilities only.

    # crxTestingSetOutputDecFunc = logReg.decision_function(inputFeatures)

    # Print metric info: Classification Accuracy, Balanced Accuracy, Confusion Matrix, PR curve, Average Precision.
    printMetricInfo(expectedOutput, crxTestingSetOutputHat,
                    crxTestingSetOutputPredictProba, dataSetType, penalty, class_weight)


# Print metric information for a classification result.
# Specifically, show: Classification Accuracy, Balanced Accuracy, Confusion Matrix, PR curve, Average Precision.
#
# Params: y - Expected output.
#         yHat - Predicted classes from a classifier.
#         yHatProba - Predicted class probabilities from a classifier.
#         dataSetType - Descriptor for the data set (i.e., 'Testing' or 'Training').
#         penalty - What penalty parameter was used with the classifier.
#         class_weight - What class_weight parameter was used with the classifier.
def printMetricInfo(y, yHat, yHatProba, dataSetType, penalty, class_weight):
    # Set identifier string for titles and descriptive console output.
    identificationString = dataSetType + ' Set - Logistic Regression (penalty=\'' + penalty + '\', class_weight=\'' + class_weight + '\'): '
    print(identificationString)

    # Show the Classification Accuracy:
    accuracyScore = accuracy_score(y, yHat)
    print('Classification Accuracy = {0:0.3f}'
          .format(accuracyScore))

    # Show the Balanced Accuracy:
    balancedAccuracyScore = balanced_accuracy_score(y, yHat)
    print('Balanced Accuracy = {0:0.3f}'
          .format(balancedAccuracyScore))

    # Show the Confusion Matrix:
    confusionMatrix = confusion_matrix(y, yHat)
    print('Confusion Matrix = \n',
          confusionMatrix)

    # Show the Average Precision Score:
    averagePrecision = average_precision_score(y, yHatProba)
    print('Average Precision Score = {0:0.3f}'
          .format(averagePrecision))

    # Show the Precision-Recall Curve:
    precision, recall, thresholds = precision_recall_curve(y, yHatProba)
    title = identificationString + '\nPrecision-Recall Curve:'
    pyplot.figure(figsize=(8, 6))
    pyplot.plot(recall, precision, marker='.')
    pyplot.title(title + "\nAverage Precision = {0:0.3f}".format(averagePrecision))
    pyplot.xlabel('Recall (Positive Label: 1)')
    pyplot.ylabel('Precision (Positive Label: 1)')
    pyplot.show()
    # pyplot.savefig(dataSetType + '_' + penalty + '_' + class_weight + '.png', dpi=300)  # Save plots for report.

    # Confirms Average Precision is the area under the Precision-Recall Curve (not exactly for sklearn implementation).
    # print("PR Curve AUC = {0:0.3f}".format(auc(recall, precision)))

    print("")  # New line for better output quality.


# Process the data by scaling.
# This method was created for the advanced tasks so as to not interrupt the implementation for parts 1-3.
#
# Params: crxInputSet - The input data set to be processed.
#         crxOutputSet - The output data set to be processed.
#         scaler - StandardScaler to use for standardisation. 'None' when processing training, given for testing.
#
# Return: The processed input set, the processed output set, and the scaler.
def processDataAdv(crxInputSet, crxOutputSet, scaler):
    # Data Standardisation:
    if scaler is None:
        # If training set, then fit.
        scaler = StandardScaler().fit(crxInputSet)

    # If testing set, then use the StandardScaler from training for the transform.
    crxInputSet_Standardised = scaler.transform(crxInputSet)

    # Convert to DataFrame for ease.
    crxInputSet_Standardised = pd.DataFrame(crxInputSet_Standardised, columns=crxInputSet.columns)

    return crxInputSet_Standardised, crxOutputSet, scaler
