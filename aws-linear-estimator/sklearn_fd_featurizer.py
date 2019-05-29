from __future__ import print_function

import time
import sys
from io import StringIO
import os
import shutil

import argparse
import csv
import json
import numpy as np
import pandas as pd

#from future_encoders import ColumnTransformer
from sklearn.compose import ColumnTransformer
from sklearn.externals import joblib
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Binarizer, StandardScaler, OneHotEncoder

from sagemaker_containers.beta.framework import (content_types, encoders, env, modules, transformer, worker)

# Since we get a headerless CSV file we specify the column names here.
# feature_columns_names = data.loc[:, data.columns != 'Class'].columns.tolist()

# label_column = 'Class'

# feature_columns_dtype = data.loc[:, data.columns != 'Class'].dtypes.to_dict()

# label_column_dtype = {'Class': np.int64} # +1.5 gives the age in years

def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    args = parser.parse_args()

    # Take the set of files and read them all into a single pandas dataframe
    input_files = [ os.path.join(args.train, file) for file in os.listdir(args.train) ]
    if len(input_files) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(args.train, "train"))
    
#     raw_data = [ pd.read_csv(
#         file, 
#         header=None, 
#         names=feature_columns_names + [label_column],
#         dtype=merge_two_dicts(feature_columns_dtype, label_column_dtype)) for file in input_files ]
#     concat_data = pd.concat(raw_data)
    print(input_files)
    raw_data = [pd.read_csv(file) for file in input_files]
    concat_data = pd.concat(raw_data)
    # This section is adapted from the scikit-learn example of using preprocessing pipelines:
    #
    # https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html
    #
    # We will train our classifier with the following features:
    # Numeric Features:
    # - length:  Longest shell measurement
    # - diameter: Diameter perpendicular to length
    # - height:  Height with meat in shell
    # - whole_weight: Weight of whole abalone
    # - shucked_weight: Weight of meat
    # - viscera_weight: Gut weight (after bleeding)
    # - shell_weight: Weight after being dried
    # Categorical Features:
    # - sex: categories encoded as strings {'M', 'F', 'I'} where 'I' is Infant
        
    feature_columns_names = concat_data.loc[:, concat_data.columns != 'Class'].columns.tolist()
    numeric_features = list(feature_columns_names)
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    preprocessor = ColumnTransformer( transformers=[ ('num', numeric_transformer, numeric_features)], remainder="drop")
    
    preprocessor.fit(concat_data)

    joblib.dump(preprocessor, os.path.join(args.model_dir, "model.joblib"))

    print("saved model!")
    
    
def input_fn(input_data, content_type):
    """Parse input data payload
    
    
    We currently only take csv input. Since we need to process both labelled
    and unlabelled data we first determine whether the label column is present
    by looking at how many columns were provided.
    """

    feature_columns_names= ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10','V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
    if content_type == 'text/csv':
        # Read the raw input data as CSV.
        df = pd.read_csv(StringIO(input_data), header = None)

        #df.columns = feature_columns_names
        if len(df.columns) > len(feature_columns_names):
            # This is a labelled example, includes the class label
            df.columns = feature_columns_names + ['Class']
        elif len(df.columns) == len(feature_columns_names):
            # This is an unlabelled example.
            df.columns = feature_columns_names
  
        return df
    else:
        raise ValueError("{} not supported by script!".format(content_type))
        

def output_fn(prediction, accept):
    """Format prediction output
    
    The default accept/content-type between containers for serial inference is JSON.
    We also want to set the ContentType or mimetype as the same value as accept so the next
    container can read the response payload correctly.
    """
    if accept == "application/json":
        instances = []
        for row in prediction.tolist():
            instances.append({"features": row})

        json_output = {"instances": instances}

        return worker.Response(json.dumps(json_output), mimetype=accept)
    elif accept == 'text/csv':
        return worker.Response(encoders.encode(prediction, accept), mimetype=accept)
    else:
        raise RuntimeException("{} accept type is not supported by this script.".format(accept))


def predict_fn(input_data, model):
    """Preprocess input data
    
    We implement this because the default predict_fn uses .predict(), but our model is a preprocessor
    so we want to use .transform().

    The output is returned in the following order:
    
        rest of features either one hot encoded or standardized
    """

    if 'Class' in input_data:
        
        features = model.transform(input_data.loc[:, input_data.columns != 'Class'])
        # Return the label (as the first column) and the set of features.
        return np.insert(features, 0, input_data['Class'], axis=1)
    else:
        
        # Return only the set of features
        features = model.transform(input_data)
        return features
    

def model_fn(model_dir):
    """Deserialize fitted model
    """
    preprocessor = joblib.load(os.path.join(model_dir, "model.joblib"))
    return preprocessor