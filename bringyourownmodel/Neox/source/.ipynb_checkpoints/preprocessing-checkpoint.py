import pandas as pd
import numpy as np
import argparse
import os
from sklearn.preprocessing import OrdinalEncoder

from datasets import load_from_disk


def _parse_args():

    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    parser.add_argument('--filepath', type=str, default='/opt/ml/processing/input/')
    parser.add_argument('--outputpath', type=str, default='/opt/ml/processing/output/')

    return parser.parse_known_args()

    
if __name__=="__main__":
    # Process arguments
    args, _ = _parse_args()
    
    
    # Load data
    data = load_from_disk(args.filepath)
    
    # Processing logic
    train_data = data
    
    # Save data
    train_data.save_to_disk(args.outputpath)
     
    print("## Processing complete. Exiting.")