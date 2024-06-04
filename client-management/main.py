import numpy as np
import pandas as pd
import recordlinkage

from recordlinkage.datasets import *


def main():
    pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_rows', None)

    df_train = load_febrl2(return_links=True)[0]  # DataFrame to train
    df_test = load_febrl1(return_links=True)[0]  # DataFrame to test
    y_train = load_febrl2(return_links=True)[1]  # True links to train
    y_test = load_febrl1(return_links=True)[1]  # True links to test

    indexer = recordlinkage.Index().full()  # Initiate indexer
    candidate_links = indexer.index(df_train)  # Index pairs of records for the training set
    candidate_links_test = indexer.index(df_test)  # Index pairs of records for the testing set

    comp = recordlinkage.Compare()  # Initiate the compare module

    # Set up thresholds for the comparison process
    comp.string(
        "given_name",
        "given_name",
        method="jarowinkler",
        label="given_name",
        threshold=0.75
    )
    comp.string(
        "surname",
        "surname",
        method="jarowinkler",
        label="surname",
        threshold=0.75
    )
    comp.string("street_number", "street_number", label="street_number", threshold=0.85)
    comp.string("address_1", "address_1", label="address_1", threshold=0.85)
    comp.string("address_2", "address_2", label="address_2", threshold=0.85)
    comp.string("suburb", "suburb", label="suburb", threshold=0.85)
    comp.string("postcode", "postcode", label="postcode", threshold=0.85)
    comp.string("date_of_birth", "date_of_birth", label="date_of_birth", threshold=0.85)
    comp.exact("soc_sec_id", "soc_sec_id", label="soc_sec_id")

    # Start comparing
    features = comp.compute(candidate_links, df_train)
    features_test = comp.compute(candidate_links_test, df_test)

    # Classification
    matches = features[features.sum(axis=1) > 3]

    # Display classification quality
    print(recordlinkage.precision(y_train, matches))
    print(recordlinkage.recall(y_train, matches))
    print(recordlinkage.fscore(y_train, matches))

    # Check algorithm's matching vs. true links
    golden_pairs = features
    golden_matches_index = golden_pairs.index.intersection(y_train)

    log_reg = recordlinkage.LogisticRegressionClassifier()  # Initiate logistic regression classifier

    log_reg.fit(golden_pairs, golden_matches_index)  # Train data
    log_reg_predict = log_reg.predict(features)  # Predict data

    # Display the quality of logistic regression
    print(recordlinkage.precision(y_train, log_reg_predict))
    print(recordlinkage.recall(y_train, log_reg_predict))
    print(recordlinkage.fscore(y_train, log_reg_predict))

    log_reg_predict_test = log_reg.predict(features_test)  # Predict based on the trained data

    # Display quality
    print(recordlinkage.precision(y_test, log_reg_predict_test))
    print(recordlinkage.recall(y_test, log_reg_predict_test))
    print(recordlinkage.fscore(y_test, log_reg_predict_test))


if __name__ == "__main__":
    main()
