print('Importing packages')

import pandas as pd
import pickle as pkl
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import chi2_contingency
import lightgbm as lgb
import datetime
import os

def preprocess(df):
    """Preprocessing logic for the chunk of data."""
    # Example preprocessing logic
    tsIdTr = df.copy()  ###################################################################
    print(tsIdTr)

    print(os.getcwd())

    # Load the preprocess_info dictionary
    with open('/home/watoomi/SafeBank/2-data-stream/c-preprocess-classify/preprocess_info/preprocess_info.pkl', 'rb') as file:
        preprocess_info = pkl.load(file)

    ## ONCE OFF: rename the columns 

    # Identify columns in tsIdTr that need renaming (those with hyphens)
    mismatched_columns = {'id-22', 'id-16', 'id-17', 'id-13', 'id-20', 'id-34', 'id-03', 'id-21', 'id-01', 'id-06', 'id-27', 'id-02', 'id-26', 
                        'id-14', 'id-35', 'id-10', 'id-05', 'id-33', 'id-31', 'id-30', 'id-12', 'id-19', 'id-23', 'id-36', 'id-38', 'id-37', 
                        'id-07', 'id-24', 'id-11', 'id-15', 'id-25', 'id-32', 'id-09', 'id-18', 'id-08', 'id-29', 'id-28', 'id-04'}


    # Create a mapping of old column names to new column names
    rename_mapping = {col: col.replace('-', '_') for col in mismatched_columns}

    # Rename columns in tsIdTr
    tsIdTr.rename(columns=rename_mapping, inplace=True)

    # Verify the renaming
    print("Updated columns in tsIdTr:", list(tsIdTr.columns))

    # Convert both lists to sets
    set_tsIdTr = set(tsIdTr.columns)
    set_loaded_columns = set(preprocess_info['trIdTr_initial_columns'])

    # Find columns in trIdTr but not in loaded_columns
    missing_in_loaded = set_tsIdTr - set_loaded_columns

    # Find columns in loaded_columns but not in tsIdTr
    missing_in_tsIdTr = set_loaded_columns - set_tsIdTr

    # Print results
    print("Columns in tsIdTr but not in loaded_columns:", missing_in_loaded)
    print("Columns in loaded_columns but not in tsIdTr:", missing_in_tsIdTr)
    print('All sets exclude isFraud, for testing')

    ############ NEEDED FOR test data ##########################################
    # Load the preprocess_info dictionary (assumes it's already loaded)
    # Load all necessary variables from preprocess_info
    categorical_features_base = preprocess_info['categorical_features']
    categorical_features = categorical_features_base.copy()

    numerical_features_base = preprocess_info['numerical_features']
    numerical_features = numerical_features_base.copy()

    label_encoders = preprocess_info['label_encoders']
    high_cardinality_features = preprocess_info['high_cardinality_features']
    low_cardinality_features = preprocess_info['low_cardinality_features']
    target_encodings = preprocess_info['target_encodings']
    column_types = preprocess_info['column_types']

    ############## EXPERIMENTATION - verify that label encoders work correctly

    feature = 'ProductCD'

    # Get the label encoder for the feature
    le = label_encoders.get(feature)

    if le:
        print(f"Classes for {feature}: {le.classes_}")
        print(f"Mapping for {feature}: {dict(zip(le.classes_, range(len(le.classes_))))}")
    else:
        print(f"Label encoder for {feature} not found.")


    # prepare and save data types of each column: 
    # Assuming column_types is the dictionary from the preprocess_info_to_update['column_types']
    for col, dtype in column_types.items():
        if col in tsIdTr.columns:
            tsIdTr[col] = tsIdTr[col].astype(dtype)

    # Check if the column types have been updated
    print(tsIdTr.dtypes)

    print('type of dataframe is ' + str(type(tsIdTr)))

    # print('saving to csv')
    # tsIdTr.to_csv('tsIdTr_ex.csv')

    ## imprvoed time complexity compared to beforehand 

    # Dictionary to track features with new unseen classes
    new_classes_tracker = {}

    # Apply label encoding to the test data
    for col in low_cardinality_features:
        le = label_encoders.get(col)  # Load the pre-saved encoder from preprocess_info
        if le:
            print(f"Applying label encoding to feature: {col}")

            # Convert column values to strings for compatibility with the label encoder
            col_values = tsIdTr[col].astype(str)  # Ensure values are strings
            
            # Track unseen labels
            unseen_labels = set(col_values) - set(le.classes_)
            if unseen_labels:
                print(f"Unseen labels in {col}: {unseen_labels}")
                new_classes_tracker[col] = list(unseen_labels)
            
            # Append unseen labels to the encoder's classes
            le_classes = np.array(list(le.classes_))  # Convert the classes to a NumPy array
            all_classes = np.concatenate([le_classes, list(unseen_labels)])  # Concatenate as arrays
            
            # Rebuild the label encoder with all classes
            le.classes_ = all_classes
            
            # Perform label encoding with unseen labels mapped to -1
            tsIdTr[col] = col_values.apply(
                lambda x: le.transform([x])[0] if x in le_classes else -1
            )


    # # Dictionary to track features with new unseen classes
    # new_classes_tracker = {}

    # # Apply label encoding to the test data
    # for col in low_cardinality_features:
    #     le = label_encoders.get(col)  # Load the pre-saved encoder from preprocess_info
    #     if le:
    #         print(f"Applying label encoding to feature: {col}")

    #         # Convert column values to strings for compatibility with the label encoder
    #         col_values = tsIdTr[col].astype(str) # .tolist()  Convert to a list, avoiding NumPy array

    #         # Track unseen labels
    #         unseen_labels = set(col_values) - set(le.classes_)
    #         if unseen_labels:
    #             print(f"Unseen labels in {col}: {unseen_labels}")
    #             new_classes_tracker[col] = list(unseen_labels)
            
    #         # Append unseen labels to the encoder's classes
    #         le_classes = list(le.classes_)  # Convert the classes to a list
    #         all_classes = le_classes + list(unseen_labels)  # Concatenate as lists
            
    #         # Rebuild the label encoder with all classes
    #         le.classes_ = all_classes
            
    #         # Perform label encoding with unseen labels mapped to -1
    #         tsIdTr[col] = tsIdTr[col].apply(lambda x: le.transform([x])[0] if x in le_classes else -1)

    # # Dictionary to track features with new unseen classes
    # new_classes_tracker = {}

    # # Apply label encoding to the test data
    # for col in low_cardinality_features:
    #     le = label_encoders.get(col)  # Load the pre-saved encoder from preprocess_info
    #     if le:
    #         print(f"Applying label encoding to feature: {col}")

    #         # Convert column values to strings for compatibility with the label encoder
    #         col_values = tsIdTr[col].astype(str).values

    #         # Track unseen labels
    #         unseen_labels = set(col_values) - set(le.classes_)
    #         if unseen_labels:
    #             print(f"Unseen labels in {col}: {unseen_labels}")
    #             new_classes_tracker[col] = list(unseen_labels)
            
    #         # Append unseen labels to the encoder's classes
    #         le_classes = np.array(le.classes_)
    #         all_classes = np.concatenate([le_classes, list(unseen_labels)])
            
    #         # Rebuild the label encoder with all classes
    #         le.classes_ = all_classes
            
    #         # Perform label encoding with unseen labels mapped to -1
    #         tsIdTr[col] = np.where(
    #             np.isin(col_values, le_classes),
    #             le.transform(col_values),
    #             -1
    #         )

    ############## EXPERIMENTATION - verify that label encoders work correctly - finding out unseen labels in classes
    classes_with_unseen = ['id_18','id_24','id_32']

    for col in classes_with_unseen:
        feature = col

        # Get the label encoder for the feature
        le = label_encoders.get(feature)

        if le:
            print(f"Classes for {feature}: {le.classes_}")
            print(f"Mapping for {feature}: {dict(zip(le.classes_, range(len(le.classes_))))}")
        else:
            print(f"Label encoder for {feature} not found.")

    # Dictionary to track new classes for target encoding (in case there are unseen categories in the test data)
    new_target_classes_tracker = {}

    # Apply target encoding for high-cardinality features
    for col in high_cardinality_features:
        if col in target_encodings:
            encoding_map = target_encodings[col]
            
            # Check for unseen categories in the test set
            unseen_categories = set(tsIdTr[col].unique()) - set(encoding_map.keys())
            if unseen_categories:
                print(f"Unseen categories in {col}: {unseen_categories}")
                new_target_classes_tracker[col] = list(unseen_categories)
            
            # Compute the target encoding (mean of the target variable for each category)
            # We assume target encoding was computed based on the training set (hence no 'isFraud' in the test set)
            # Use the original encoding_map and apply the target encoding to the test data.
            
            # If you have the target encoding directly as a dictionary, apply it:
            tsIdTr[f'{col}_target_enc'] = tsIdTr[col].map(encoding_map)

            # Handle missing categories or those not present in the encoding map
            tsIdTr[f'{col}_target_enc'].fillna(-1, inplace=True)  # Use -1 for unseen categories (you can also use NaN)

    print("Target encoding applied to test data for high-cardinality features:", high_cardinality_features)

    ############ NEEDED FOR test data ######################################################

    ### add in TransactionDT enriched columns:

    ### numerical 
    tsIdTr['TransactionDT_days'] = tsIdTr['TransactionDT'] // (24 * 60 * 60)  # Convert to days 

    ### categorical

    tsIdTr['TransactionDT_weekday'] = tsIdTr['TransactionDT_days'] % 7  # Day of the week (0-6)
    tsIdTr['TransactionDT_hour'] = (tsIdTr['TransactionDT']//3600) % 24  # Calculate hour of the day (0-23)

    ### numerical
    tsIdTr['TransactionDT_hours'] = (tsIdTr['TransactionDT'] // (60 * 60)) % 24  # Extract hours - hours since reference point in dataset

    tsIdTr['hour_sin'] = np.sin(2 * np.pi * tsIdTr['TransactionDT_hour'] / 24)
    tsIdTr['hour_cos'] = np.cos(2 * np.pi * tsIdTr['TransactionDT_hour'] / 24)
    tsIdTr['weekday_sin'] = np.sin(2 * np.pi * tsIdTr['TransactionDT_weekday'] / 7)
    tsIdTr['weekday_cos'] = np.cos(2 * np.pi * tsIdTr['TransactionDT_weekday'] / 7)

    ####    TransactionAmt split into whole dollars and cents (so one dollars/whole number, and one cents column)
    tsIdTr['TransactionAmt_dollars'] = tsIdTr['TransactionAmt'] // 1  # Extract whole dollars
    tsIdTr['TransactionAmt_cents'] = (tsIdTr['TransactionAmt'] * 100) % 100  # Extract cents

    # update the feature sets: 
    ### categorical_features
    ### numerical_features , 

    # Update categorical_features with newly created categorical columns
    categorical_features += [
        'TransactionDT_weekday', 'TransactionDT_hour',  # New categorical features
    ]

    # Update numerical_features with newly created numerical columns
    numerical_features += [
        'TransactionDT_days', 'TransactionDT_hours',  # Existing numerical features
        'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos',  # Cyclical features
        'TransactionAmt_dollars', 'TransactionAmt_cents',  # New features from TransactionAmt
        'DeviceInfo_target_enc', 'id_30_target_enc', 'id_25_target_enc', 'id_31_target_enc', 
        'id_20_target_enc', 'addr2_target_enc', 'card5_target_enc', 'card2_target_enc', 
        'addr1_target_enc', 'id_13_target_enc', 'id_19_target_enc', 'id_21_target_enc', 
        'card3_target_enc', 'id_17_target_enc', 'P_emaildomain_target_enc', 'id_33_target_enc', 
        'id_22_target_enc', 'id_26_target_enc', 'R_emaildomain_target_enc', 'card1_target_enc',
        'DeviceInfo_target_enc', 'id_30_target_enc', 'id_25_target_enc', 'id_31_target_enc', 
        'id_20_target_enc', 'addr2_target_enc', 'card5_target_enc', 'card2_target_enc', 
        'addr1_target_enc', 'id_13_target_enc', 'id_19_target_enc', 'id_21_target_enc', 
        'card3_target_enc', 'id_17_target_enc', 'P_emaildomain_target_enc', 'id_33_target_enc', 
        'id_22_target_enc', 'id_26_target_enc', 'R_emaildomain_target_enc', 'card1_target_enc', 
        'id_14_target_enc'
    ]

    # Check the final lists
    print("Updated categorical features:", categorical_features)
    print("Updated numerical features:", numerical_features)

    ### note TransactionDT_hours,TransactionDT_days, are not important features so should not be put into either numieracl or categorerial (but still remain in dataset)

    ### find set which are not in either

    # Get all columns from the dataframe
    all_columns = set(tsIdTr.columns)

    # Combine categorical and numerical features
    all_features = set(categorical_features + numerical_features)

    # Identify features that are not in either categorical or numerical lists
    missing_features = all_columns - all_features

    # Print the missing features
    print("Features not included in either categorical or numerical features:", missing_features)

    # check that all columns are in categorical_features or numerical_features; if not, remove column, print removed column names. 
    # exclusions are: 'Unnamed: 0', 'isFraud', 'TransactionID' #

    # Remove target-encoded features from categorical_features if they exist in numerical_features (target encoded have become numerical now)
    categorical_features = [col for col in categorical_features if col + '_target_enc' not in numerical_features]

    # Remove TransactionDT_hours and TransactionDT_days from numerical_features
    numerical_features = [col for col in numerical_features if col not in ['TransactionDT_hours', 'TransactionDT_days']]

    # Create a set of all columns that should be included
    all_columns = set(categorical_features + numerical_features)

    # Exclude specific columns from being considered
    excluded_columns = {'TransactionID'} # 'isFraud', here usually but excluded for testing

    # Remove any columns from the dataframe that are not in the updated feature sets
    removed_columns = []
    for col in tsIdTr.columns:
        if col not in all_columns and col not in excluded_columns:
            removed_columns.append(col)
            tsIdTr.drop(col, axis=1, inplace=True)

    # Print the names of removed columns
    print("Removed columns:", removed_columns)

    # Verify that all remaining columns are accounted for
    remaining_columns = set(tsIdTr.columns)
    assert all(col in all_columns or col in excluded_columns for col in remaining_columns), "Some columns are not in either feature list!"

    # Remove 'TransactionDT' and 'TransactionAmt' from numerical_features
    numerical_features = [col for col in numerical_features if col not in ['TransactionDT', 'TransactionAmt']]

    # Drop 'TransactionDT' and 'TransactionAmt' from the DataFrame
    tsIdTr.drop(['TransactionDT', 'TransactionAmt'], axis=1, inplace=True)

    # create copy
    tsIdTr2 = tsIdTr.copy()

    low_correlation_features = preprocess_info['low_correlation_features']

    ########## duplicated here: 
    # Remove low correlation features from the test set (tsIdTr)
    tsIdTr2 = tsIdTr2.drop(columns=low_correlation_features, errors='ignore')

    # Remove features with low correlation
    numerical_features = [feature for feature in numerical_features if feature not in low_correlation_features]

    ## update tsIdTr2 to include features only needed:

    # Define the columns to include
    included_columns = [col for col in numerical_features + categorical_features if col != 'isFraud']

    # 'Unnamed: 0',

    # Update tsIdTr2 by selecting only the desired columns
    tsIdTr2 = tsIdTr2[included_columns]

    # Print the shape of the updated dataframe
    print(f"Updated dataframe shape: {tsIdTr2.shape}")

    #create copy
    tsIdTr3 = tsIdTr2.copy()

    print(tsIdTr3.shape)

    # Identify duplicated columns
    duplicate_columns = tsIdTr3.columns[tsIdTr3.columns.duplicated()]

    # Log the duplicated columns
    print(f"Duplicated columns: {duplicate_columns.tolist()}")

    # Drop duplicate columns
    tsIdTr3 = tsIdTr3.loc[:, ~tsIdTr3.columns.duplicated()]

    # Print the shape of the updated dataframe
    print(f"Updated dataframe shape: {tsIdTr3.shape}")

    ######################### Save shapes before and after preprocessing ##########################
    # Training
    train_before_shape = preprocess_info['train_before_shape'] 
    train_after_shape = preprocess_info['train_after_shape'] 

    # Test
    test_before_shape = tsIdTr.shape
    test_after_shape = tsIdTr3.shape

    ######################### Save columns before and after preprocessing ##########################

    # Save columns before and after preprocessing for training
    train_columns_before = preprocess_info['train_columns_before'] 
    train_columns_after = preprocess_info['train_columns_after']

    # Save columns before and after preprocessing for test
    test_columns_before = tsIdTr.columns.tolist()
    test_columns_after = tsIdTr3.columns.tolist()

    ######################### Check differences in columns ##########################
    # For training
    train_columns_removed = [col for col in preprocess_info['train_columns_before'] if col not in preprocess_info['train_columns_after']]
    train_columns_added = [col for col in preprocess_info['train_columns_after'] if col not in preprocess_info['train_columns_before']]

    # For test (excluding 'isFraud' in the test set)
    test_columns_removed = [col for col in test_columns_before if col not in test_columns_after and col != 'isFraud']
    test_columns_added = [col for col in test_columns_after if col not in test_columns_before and col != 'isFraud']

    # Save the differences to the dictionary
    train_columns_removed = preprocess_info['train_columns_removed'] 
    train_columns_added = preprocess_info['train_columns_added']

    ######################### Handle isFraud in the test set ##########################
    # If 'isFraud' is in the test columns after preprocessing, remove it (since it shouldn't be in the test set)
    if 'isFraud' in test_columns_after:
        #### ASSERT IS THERE THE isFraud column  
        test_columns_after.remove('isFraud')

    ######################### Print shapes before and after preprocessing ##########################
    print("### Training Data Shapes ###")
    print(f"Training data before preprocessing: {train_before_shape}")
    print(f"Training data after preprocessing: {train_after_shape}\n")

    print("### Test Data Shapes ###")
    print(f"Test data before preprocessing: {test_before_shape}")
    print(f"Test data after preprocessing: {test_after_shape}\n")

    ######################### Print columns before and after preprocessing ##########################
    print("### Training Data Columns ###")
    print(f"Columns in training data before preprocessing: {len(train_columns_before)}")
    print(f"Columns in training data after preprocessing: {len(train_columns_after)}\n")

    print("### Test Data Columns ###")
    print(f"Columns in test data before preprocessing: {len(test_columns_before)}")
    print(f"Columns in test data after preprocessing: {len(test_columns_after)}\n")

    ######################### Print differences in columns ##########################
    # For training data
    print("### Training Data Column Differences ###")
    print(f"Columns removed from training data: {len(train_columns_removed)}")
    print(f"Columns added to training data: {len(train_columns_added)}\n")

    # For test data
    print("### Test Data Column Differences ###")
    print(f"Columns removed from test data: {len(test_columns_removed)}")
    print(f"Columns added to test data: {len(test_columns_added)}\n")

    # Display removed and added columns for both training and test data if needed
    if train_columns_removed:
        print(f"Removed columns in training data: {train_columns_removed}")
    if train_columns_added:
        print(f"Added columns in training data: {train_columns_added}")

    if test_columns_removed:
        print(f"Removed columns in test data: {test_columns_removed}")
    if test_columns_added:
        print(f"Added columns in test data: {test_columns_added}")


    # train_columns_removed = train_columns_removed.remove('TransactionID')
    # # train_columns_removed = train_columns_removed.remove('TransactionID')

    cols_remove = [col for col in train_columns_removed if col != 'TransactionID']

    tsIdTr4 = tsIdTr3.drop(columns=cols_remove) 
    print('tsIdTr4 len is: '+ str(tsIdTr4.shape))

    # Prepare the test features (excluding the 'isFraud' column for prediction)
    X_test = tsIdTr4.drop(columns=['TransactionID']) 

    # Step 2: Load the pre-trained LightGBM model
    model_path = '/home/watoomi/SafeBank/2-data-stream/c-preprocess-classify//preprocess_info/safeBank_lightGBM_model.txt'
    lgbm_model = lgb.Booster(model_file=model_path)

    # Get the feature names the model was trained on
    expected_columns = lgbm_model.feature_name()
    # Print the expected feature names
    print(f"Expected features in the model: {expected_columns}")
    print(f'len of expected columns: : {len(expected_columns)}')

    # Get the feature names the model was trained on
    test_columns = list(X_test.columns)
    # Print the expected feature names
    print(f"Expected features in the model: {test_columns}")
    print(f'len of expected columns: : {len(test_columns)}')

    # Find the columns in the test set that are not expected by the model
    columns_to_remove = [col for col in test_columns if col not in expected_columns]

    # Print the columns removed and the remaining ones
    print(f"Removed columns: {columns_to_remove}")

    X_test_reduced = X_test.drop(columns=columns_to_remove)

    # Get the feature names the model was trained on
    expected_columns = lgbm_model.feature_name()
    # Print the expected feature names
    print(f"Expected features in the model: {expected_columns}")
    print(f'len of expected columns: : {len(expected_columns)}')

    # Get the feature names the model was trained on
    test_columns = list(X_test_reduced.columns)
    # Print the expected feature names
    print(f"Expected features in the model: {test_columns}")
    print(f'len of expected columns: : {len(test_columns)}')

    # Find the columns in the test set that are not expected by the model
    columns_to_remove = [col for col in expected_columns if col not in test_columns]

    # Print the columns removed and the remaining ones
    print(f"Removed columns: {columns_to_remove}")


    # Option 2: Add missing columns to the test set (fill with NaN or some default value)
    for col in columns_to_remove:
        X_test_reduced[col] = np.nan  # or some placeholder value if needed

    # Step 3: Perform predictions on the test set
    predictions = lgbm_model.predict(X_test_reduced)
    # Probability of fraud
    probability_predictions = predictions  # The raw output probabilities from the model

    # Binary predictions with a custom threshold
    binary_predictions = (probability_predictions > 0.7).astype(int)

    # Get current time in the format YYYY-MM-DD_HH-MM-SS
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    timestamp_short = datetime.datetime.now().strftime('%Y-%m-%d')

    # Collect prediction metadata (add more as needed)
    prediction_metadata = {
        'model_version': 'v_Mod'+ timestamp_short, # after hard encoded
        'preprocessing_version': 'v_Pro'+ timestamp_short,
        'prediction_threshold': 0.7,  # The threshold for binary classification
        'classification_time': timestamp,
        'model_name': 'LightGBM Fraud Detection',  # Optional model name
        'features_used': X_test_reduced.columns.tolist(),  # Columns/features used in prediction
    }

    # Step 4: Prepare the DataFrame to store the predictions along with additional columns
    predictions_df = pd.DataFrame({
        'TransactionID': tsIdTr4['TransactionID'],
        'isFraud_Prediction': binary_predictions,
        'Fraud_Probability': probability_predictions,  # Probability of fraud
        'Classification_Time': timestamp,  # Time of classification
        'Model_Version': prediction_metadata['model_version'],
        'Prediction_Threshold': prediction_metadata['prediction_threshold'],
        'Model_Name': prediction_metadata['model_name']
    })

    return predictions_df



#### OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD OLD 
# def preprocess(df):
#     """Preprocessing logic for the chunk of data."""
#     # Example preprocessing logic
#     tsIdTr = df.copy()
#     print(tsIdTr)

#     print(os.getcwd())

#     # Load the preprocess_info dictionary
#     with open('2-data-stream/c-preprocess-classify/preprocess_info/preprocess_info.pkl', 'rb') as file:
#         preprocess_info = pkl.load(file)

#     ## ONCE OFF: rename the columns 

#     # Identify columns in tsIdTr that need renaming (those with hyphens)
#     mismatched_columns = {'id-22', 'id-16', 'id-17', 'id-13', 'id-20', 'id-34', 'id-03', 'id-21', 'id-01', 'id-06', 'id-27', 'id-02', 'id-26', 
#                         'id-14', 'id-35', 'id-10', 'id-05', 'id-33', 'id-31', 'id-30', 'id-12', 'id-19', 'id-23', 'id-36', 'id-38', 'id-37', 
#                         'id-07', 'id-24', 'id-11', 'id-15', 'id-25', 'id-32', 'id-09', 'id-18', 'id-08', 'id-29', 'id-28', 'id-04'}


#     # Create a mapping of old column names to new column names
#     rename_mapping = {col: col.replace('-', '_') for col in mismatched_columns}

#     # Rename columns in tsIdTr
#     tsIdTr.rename(columns=rename_mapping, inplace=True)

#     # Verify the renaming
#     print("Updated columns in tsIdTr:", list(tsIdTr.columns))

#     # Convert both lists to sets
#     set_tsIdTr = set(tsIdTr.columns)
#     set_loaded_columns = set(preprocess_info['trIdTr_initial_columns'])

#     # Find columns in trIdTr but not in loaded_columns
#     missing_in_loaded = set_tsIdTr - set_loaded_columns

#     # Find columns in loaded_columns but not in tsIdTr
#     missing_in_tsIdTr = set_loaded_columns - set_tsIdTr

#     # Print results
#     print("Columns in tsIdTr but not in loaded_columns:", missing_in_loaded)
#     print("Columns in loaded_columns but not in tsIdTr:", missing_in_tsIdTr)
#     print('All sets exclude isFraud, for testing')

#     ############ NEEDED FOR test data ##########################################
#     # Load the preprocess_info dictionary (assumes it's already loaded)
#     # Load all necessary variables from preprocess_info
#     categorical_features_base = preprocess_info['categorical_features']
#     categorical_features = categorical_features_base.copy()

#     numerical_features_base = preprocess_info['numerical_features']
#     numerical_features = numerical_features_base.copy()

#     label_encoders = preprocess_info['label_encoders']
#     high_cardinality_features = preprocess_info['high_cardinality_features']
#     low_cardinality_features = preprocess_info['low_cardinality_features']
#     target_encodings = preprocess_info['target_encodings']
#     column_types = preprocess_info['column_types']

#     ############## EXPERIMENTATION - verify that label encoders work correctly

#     feature = 'ProductCD'

#     # Get the label encoder for the feature
#     le = label_encoders.get(feature)

#     if le:
#         print(f"Classes for {feature}: {le.classes_}")
#         print(f"Mapping for {feature}: {dict(zip(le.classes_, range(len(le.classes_))))}")
#     else:
#         print(f"Label encoder for {feature} not found.")


#     # prepare and save data types of each column: 
#     # Assuming column_types is the dictionary from the preprocess_info_to_update['column_types']
#     for col, dtype in column_types.items():
#         if col in tsIdTr.columns:
#             tsIdTr[col] = tsIdTr[col].astype(dtype)

#     # Check if the column types have been updated
#     print(tsIdTr.dtypes)

#     print('type of dataframe is ' + str(type(tsIdTr)))
#     print(tsIdTr.id_12.info())

#     print('saving to csv')
#     tsIdTr.to_csv('tsIdTr_ex.csv')

#     ## imprvoed time complexity compared to beforehand 

#     # Dictionary to track features with new unseen classes
#     new_classes_tracker = {}

#     # Apply label encoding to the test data
#     for col in low_cardinality_features:
#         le = label_encoders.get(col)  # Load the pre-saved encoder from preprocess_info
#         if le:
#             print(f"Applying label encoding to feature: {col}")

#             # Convert column values to strings for compatibility with the label encoder
#             col_values = tsIdTr[col].astype(str) # .tolist()  Convert to a list, avoiding NumPy array

#             # Track unseen labels
#             unseen_labels = set(col_values) - set(le.classes_)
#             if unseen_labels:
#                 print(f"Unseen labels in {col}: {unseen_labels}")
#                 new_classes_tracker[col] = list(unseen_labels)
            
#             # Append unseen labels to the encoder's classes
#             le_classes = list(le.classes_)  # Convert the classes to a list
#             all_classes = le_classes + list(unseen_labels)  # Concatenate as lists
            
#             # Rebuild the label encoder with all classes
#             le.classes_ = all_classes
            
#             # Perform label encoding with unseen labels mapped to -1
#             tsIdTr[col] = tsIdTr[col].apply(lambda x: le.transform([x])[0] if x in le_classes else -1)

#     # # Dictionary to track features with new unseen classes
#     # new_classes_tracker = {}

#     # # Apply label encoding to the test data
#     # for col in low_cardinality_features:
#     #     le = label_encoders.get(col)  # Load the pre-saved encoder from preprocess_info
#     #     if le:
#     #         print(f"Applying label encoding to feature: {col}")

#     #         # Convert column values to strings for compatibility with the label encoder
#     #         col_values = tsIdTr[col].astype(str).values

#     #         # Track unseen labels
#     #         unseen_labels = set(col_values) - set(le.classes_)
#     #         if unseen_labels:
#     #             print(f"Unseen labels in {col}: {unseen_labels}")
#     #             new_classes_tracker[col] = list(unseen_labels)
            
#     #         # Append unseen labels to the encoder's classes
#     #         le_classes = np.array(le.classes_)
#     #         all_classes = np.concatenate([le_classes, list(unseen_labels)])
            
#     #         # Rebuild the label encoder with all classes
#     #         le.classes_ = all_classes
            
#     #         # Perform label encoding with unseen labels mapped to -1
#     #         tsIdTr[col] = np.where(
#     #             np.isin(col_values, le_classes),
#     #             le.transform(col_values),
#     #             -1
#     #         )

#     ############## EXPERIMENTATION - verify that label encoders work correctly - finding out unseen labels in classes
#     classes_with_unseen = ['id_18','id_24','id_32']

#     for col in classes_with_unseen:
#         feature = col

#         # Get the label encoder for the feature
#         le = label_encoders.get(feature)

#         if le:
#             print(f"Classes for {feature}: {le.classes_}")
#             print(f"Mapping for {feature}: {dict(zip(le.classes_, range(len(le.classes_))))}")
#         else:
#             print(f"Label encoder for {feature} not found.")

#     # Dictionary to track new classes for target encoding (in case there are unseen categories in the test data)
#     new_target_classes_tracker = {}

#     # Apply target encoding for high-cardinality features
#     for col in high_cardinality_features:
#         if col in target_encodings:
#             encoding_map = target_encodings[col]
            
#             # Check for unseen categories in the test set
#             unseen_categories = set(tsIdTr[col].unique()) - set(encoding_map.keys())
#             if unseen_categories:
#                 print(f"Unseen categories in {col}: {unseen_categories}")
#                 new_target_classes_tracker[col] = list(unseen_categories)
            
#             # Compute the target encoding (mean of the target variable for each category)
#             # We assume target encoding was computed based on the training set (hence no 'isFraud' in the test set)
#             # Use the original encoding_map and apply the target encoding to the test data.
            
#             # If you have the target encoding directly as a dictionary, apply it:
#             tsIdTr[f'{col}_target_enc'] = tsIdTr[col].map(encoding_map)

#             # Handle missing categories or those not present in the encoding map
#             tsIdTr[f'{col}_target_enc'].fillna(-1, inplace=True)  # Use -1 for unseen categories (you can also use NaN)

#     print("Target encoding applied to test data for high-cardinality features:", high_cardinality_features)

#     ############ NEEDED FOR test data ######################################################

#     ### add in TransactionDT enriched columns:

#     ### numerical 
#     tsIdTr['TransactionDT_days'] = tsIdTr['TransactionDT'] // (24 * 60 * 60)  # Convert to days 

#     ### categorical

#     tsIdTr['TransactionDT_weekday'] = tsIdTr['TransactionDT_days'] % 7  # Day of the week (0-6)
#     tsIdTr['TransactionDT_hour'] = (tsIdTr['TransactionDT']//3600) % 24  # Calculate hour of the day (0-23)

#     ### numerical
#     tsIdTr['TransactionDT_hours'] = (tsIdTr['TransactionDT'] // (60 * 60)) % 24  # Extract hours - hours since reference point in dataset

#     tsIdTr['hour_sin'] = np.sin(2 * np.pi * tsIdTr['TransactionDT_hour'] / 24)
#     tsIdTr['hour_cos'] = np.cos(2 * np.pi * tsIdTr['TransactionDT_hour'] / 24)
#     tsIdTr['weekday_sin'] = np.sin(2 * np.pi * tsIdTr['TransactionDT_weekday'] / 7)
#     tsIdTr['weekday_cos'] = np.cos(2 * np.pi * tsIdTr['TransactionDT_weekday'] / 7)

#     ####    TransactionAmt split into whole dollars and cents (so one dollars/whole number, and one cents column)
#     tsIdTr['TransactionAmt_dollars'] = tsIdTr['TransactionAmt'] // 1  # Extract whole dollars
#     tsIdTr['TransactionAmt_cents'] = (tsIdTr['TransactionAmt'] * 100) % 100  # Extract cents

#     # update the feature sets: 
#     ### categorical_features
#     ### numerical_features , 

#     # Update categorical_features with newly created categorical columns
#     categorical_features += [
#         'TransactionDT_weekday', 'TransactionDT_hour',  # New categorical features
#     ]

#     # Update numerical_features with newly created numerical columns
#     numerical_features += [
#         'TransactionDT_days', 'TransactionDT_hours',  # Existing numerical features
#         'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos',  # Cyclical features
#         'TransactionAmt_dollars', 'TransactionAmt_cents',  # New features from TransactionAmt
#         'DeviceInfo_target_enc', 'id_30_target_enc', 'id_25_target_enc', 'id_31_target_enc', 
#         'id_20_target_enc', 'addr2_target_enc', 'card5_target_enc', 'card2_target_enc', 
#         'addr1_target_enc', 'id_13_target_enc', 'id_19_target_enc', 'id_21_target_enc', 
#         'card3_target_enc', 'id_17_target_enc', 'P_emaildomain_target_enc', 'id_33_target_enc', 
#         'id_22_target_enc', 'id_26_target_enc', 'R_emaildomain_target_enc', 'card1_target_enc',
#         'DeviceInfo_target_enc', 'id_30_target_enc', 'id_25_target_enc', 'id_31_target_enc', 
#         'id_20_target_enc', 'addr2_target_enc', 'card5_target_enc', 'card2_target_enc', 
#         'addr1_target_enc', 'id_13_target_enc', 'id_19_target_enc', 'id_21_target_enc', 
#         'card3_target_enc', 'id_17_target_enc', 'P_emaildomain_target_enc', 'id_33_target_enc', 
#         'id_22_target_enc', 'id_26_target_enc', 'R_emaildomain_target_enc', 'card1_target_enc', 
#         'id_14_target_enc'
#     ]

#     # Check the final lists
#     print("Updated categorical features:", categorical_features)
#     print("Updated numerical features:", numerical_features)

#     ### note TransactionDT_hours,TransactionDT_days, are not important features so should not be put into either numieracl or categorerial (but still remain in dataset)

#     ### find set which are not in either

#     # Get all columns from the dataframe
#     all_columns = set(tsIdTr.columns)

#     # Combine categorical and numerical features
#     all_features = set(categorical_features + numerical_features)

#     # Identify features that are not in either categorical or numerical lists
#     missing_features = all_columns - all_features

#     # Print the missing features
#     print("Features not included in either categorical or numerical features:", missing_features)

#     # check that all columns are in categorical_features or numerical_features; if not, remove column, print removed column names. 
#     # exclusions are: 'Unnamed: 0', 'isFraud', 'TransactionID' #

#     # Remove target-encoded features from categorical_features if they exist in numerical_features (target encoded have become numerical now)
#     categorical_features = [col for col in categorical_features if col + '_target_enc' not in numerical_features]

#     # Remove TransactionDT_hours and TransactionDT_days from numerical_features
#     numerical_features = [col for col in numerical_features if col not in ['TransactionDT_hours', 'TransactionDT_days']]

#     # Create a set of all columns that should be included
#     all_columns = set(categorical_features + numerical_features)

#     # Exclude specific columns from being considered
#     excluded_columns = {'TransactionID'} # 'isFraud', here usually but excluded for testing

#     # Remove any columns from the dataframe that are not in the updated feature sets
#     removed_columns = []
#     for col in tsIdTr.columns:
#         if col not in all_columns and col not in excluded_columns:
#             removed_columns.append(col)
#             tsIdTr.drop(col, axis=1, inplace=True)

#     # Print the names of removed columns
#     print("Removed columns:", removed_columns)

#     # Verify that all remaining columns are accounted for
#     remaining_columns = set(tsIdTr.columns)
#     assert all(col in all_columns or col in excluded_columns for col in remaining_columns), "Some columns are not in either feature list!"

#     # Remove 'TransactionDT' and 'TransactionAmt' from numerical_features
#     numerical_features = [col for col in numerical_features if col not in ['TransactionDT', 'TransactionAmt']]

#     # Drop 'TransactionDT' and 'TransactionAmt' from the DataFrame
#     tsIdTr.drop(['TransactionDT', 'TransactionAmt'], axis=1, inplace=True)

#     # create copy
#     tsIdTr2 = tsIdTr.copy()

#     low_correlation_features = preprocess_info['low_correlation_features']

#     ########## duplicated here: 
#     # Remove low correlation features from the test set (tsIdTr)
#     tsIdTr2 = tsIdTr2.drop(columns=low_correlation_features, errors='ignore')

#     # Remove features with low correlation
#     numerical_features = [feature for feature in numerical_features if feature not in low_correlation_features]

#     ## update tsIdTr2 to include features only needed:

#     # Define the columns to include
#     included_columns = [col for col in numerical_features + categorical_features if col != 'isFraud']

#     # 'Unnamed: 0',

#     # Update tsIdTr2 by selecting only the desired columns
#     tsIdTr2 = tsIdTr2[included_columns]

#     # Print the shape of the updated dataframe
#     print(f"Updated dataframe shape: {tsIdTr2.shape}")

#     #create copy
#     tsIdTr3 = tsIdTr2.copy()

#     print(tsIdTr3.shape)

#     # Identify duplicated columns
#     duplicate_columns = tsIdTr3.columns[tsIdTr3.columns.duplicated()]

#     # Log the duplicated columns
#     print(f"Duplicated columns: {duplicate_columns.tolist()}")

#     # Drop duplicate columns
#     tsIdTr3 = tsIdTr3.loc[:, ~tsIdTr3.columns.duplicated()]

#     # Print the shape of the updated dataframe
#     print(f"Updated dataframe shape: {tsIdTr3.shape}")


#     ######################### Save shapes before and after preprocessing ##########################
#     # Training
#     train_before_shape = preprocess_info['train_before_shape'] 
#     train_after_shape = preprocess_info['train_after_shape'] 

#     # Test
#     test_before_shape = tsIdTr.shape
#     test_after_shape = tsIdTr3.shape

#     ######################### Save columns before and after preprocessing ##########################

#     # Save columns before and after preprocessing for training
#     train_columns_before = preprocess_info['train_columns_before'] 
#     train_columns_after = preprocess_info['train_columns_after']

#     # Save columns before and after preprocessing for test
#     test_columns_before = tsIdTr.columns.tolist()
#     test_columns_after = tsIdTr3.columns.tolist()

#     ######################### Check differences in columns ##########################
#     # For training
#     train_columns_removed = [col for col in preprocess_info['train_columns_before'] if col not in preprocess_info['train_columns_after']]
#     train_columns_added = [col for col in preprocess_info['train_columns_after'] if col not in preprocess_info['train_columns_before']]

#     # For test (excluding 'isFraud' in the test set)
#     test_columns_removed = [col for col in test_columns_before if col not in test_columns_after and col != 'isFraud']
#     test_columns_added = [col for col in test_columns_after if col not in test_columns_before and col != 'isFraud']

#     # Save the differences to the dictionary
#     train_columns_removed = preprocess_info['train_columns_removed'] 
#     train_columns_added = preprocess_info['train_columns_added']

#     ######################### Handle isFraud in the test set ##########################
#     # If 'isFraud' is in the test columns after preprocessing, remove it (since it shouldn't be in the test set)
#     if 'isFraud' in test_columns_after:
#         #### ASSERT IS THERE THE isFraud column  
#         test_columns_after.remove('isFraud')

#     ######################### Print shapes before and after preprocessing ##########################
#     print("### Training Data Shapes ###")
#     print(f"Training data before preprocessing: {train_before_shape}")
#     print(f"Training data after preprocessing: {train_after_shape}\n")

#     print("### Test Data Shapes ###")
#     print(f"Test data before preprocessing: {test_before_shape}")
#     print(f"Test data after preprocessing: {test_after_shape}\n")

#     ######################### Print columns before and after preprocessing ##########################
#     print("### Training Data Columns ###")
#     print(f"Columns in training data before preprocessing: {len(train_columns_before)}")
#     print(f"Columns in training data after preprocessing: {len(train_columns_after)}\n")

#     print("### Test Data Columns ###")
#     print(f"Columns in test data before preprocessing: {len(test_columns_before)}")
#     print(f"Columns in test data after preprocessing: {len(test_columns_after)}\n")

#     ######################### Print differences in columns ##########################
#     # For training data
#     print("### Training Data Column Differences ###")
#     print(f"Columns removed from training data: {len(train_columns_removed)}")
#     print(f"Columns added to training data: {len(train_columns_added)}\n")

#     # For test data
#     print("### Test Data Column Differences ###")
#     print(f"Columns removed from test data: {len(test_columns_removed)}")
#     print(f"Columns added to test data: {len(test_columns_added)}\n")

#     # Display removed and added columns for both training and test data if needed
#     if train_columns_removed:
#         print(f"Removed columns in training data: {train_columns_removed}")
#     if train_columns_added:
#         print(f"Added columns in training data: {train_columns_added}")

#     if test_columns_removed:
#         print(f"Removed columns in test data: {test_columns_removed}")
#     if test_columns_added:
#         print(f"Added columns in test data: {test_columns_added}")


#     # train_columns_removed = train_columns_removed.remove('TransactionID')
#     # # train_columns_removed = train_columns_removed.remove('TransactionID')

#     cols_remove = [col for col in train_columns_removed if col != 'TransactionID']

#     tsIdTr4 = tsIdTr3.drop(columns=cols_remove) 
#     print('tsIdTr4 len is: '+ str(tsIdTr4.shape))

#     # Prepare the test features (excluding the 'isFraud' column for prediction)
#     X_test = tsIdTr4.drop(columns=['TransactionID']) 

#     # Step 2: Load the pre-trained LightGBM model
#     model_path = '../preprocess_info/safeBank_lightGBM_model.txt'
#     lgbm_model = lgb.Booster(model_file=model_path)

#     # Get the feature names the model was trained on
#     expected_columns = lgbm_model.feature_name()
#     # Print the expected feature names
#     print(f"Expected features in the model: {expected_columns}")
#     print(f'len of expected columns: : {len(expected_columns)}')

#     # Get the feature names the model was trained on
#     test_columns = list(X_test.columns)
#     # Print the expected feature names
#     print(f"Expected features in the model: {test_columns}")
#     print(f'len of expected columns: : {len(test_columns)}')

#     # Find the columns in the test set that are not expected by the model
#     columns_to_remove = [col for col in test_columns if col not in expected_columns]

#     # Print the columns removed and the remaining ones
#     print(f"Removed columns: {columns_to_remove}")

#     X_test_reduced = X_test.drop(columns=columns_to_remove)

#     # Get the feature names the model was trained on
#     expected_columns = lgbm_model.feature_name()
#     # Print the expected feature names
#     print(f"Expected features in the model: {expected_columns}")
#     print(f'len of expected columns: : {len(expected_columns)}')

#     # Get the feature names the model was trained on
#     test_columns = list(X_test_reduced.columns)
#     # Print the expected feature names
#     print(f"Expected features in the model: {test_columns}")
#     print(f'len of expected columns: : {len(test_columns)}')

#     # Find the columns in the test set that are not expected by the model
#     columns_to_remove = [col for col in expected_columns if col not in test_columns]

#     # Print the columns removed and the remaining ones
#     print(f"Removed columns: {columns_to_remove}")

#     # Option 2: Add missing columns to the test set (fill with NaN or some default value)
#     for col in columns_to_remove:
#         X_test_reduced[col] = np.nan  # or some placeholder value if needed


#     # Step 3: Perform predictions on the test set
#     predictions = lgbm_model.predict(X_test_reduced)
#     # Probability of fraud
#     probability_predictions = predictions  # The raw output probabilities from the model

#     # Binary predictions with a custom threshold
#     binary_predictions = (probability_predictions > 0.7).astype(int)

#     # Get current time in the format YYYY-MM-DD_HH-MM-SS
#     timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
#     timestamp_short = datetime.datetime.now().strftime('%Y-%m-%d')

#     # Collect prediction metadata (add more as needed)
#     prediction_metadata = {
#         'model_version': 'v_Mod'+ timestamp_short, # after hard encoded
#         'preprocessing_version': 'v_Pro'+ timestamp_short,
#         'prediction_threshold': 0.7,  # The threshold for binary classification
#         'classification_time': timestamp,
#         'model_name': 'LightGBM Fraud Detection',  # Optional model name
#         'features_used': X_test_reduced.columns.tolist(),  # Columns/features used in prediction
#     }

#     # Step 4: Prepare the DataFrame to store the predictions along with additional columns
#     predictions_df = pd.DataFrame({
#         'TransactionID': tsIdTr4['TransactionID'],
#         'isFraud_Prediction': binary_predictions,
#         'Fraud_Probability': probability_predictions,  # Probability of fraud
#         'Classification_Time': timestamp,  # Time of classification
#         'Model_Version': prediction_metadata['model_version'],
#         'Prediction_Threshold': prediction_metadata['prediction_threshold'],
#         'Model_Name': prediction_metadata['model_name']
#     })

#     return predictions_df

