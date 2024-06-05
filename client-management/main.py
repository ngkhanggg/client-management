import numpy as np
import pandas as pd
import recordlinkage

from recordlinkage.datasets import *


# Display the classification quality
def display_quality(true_links, df):
    print(recordlinkage.precision(true_links, df))
    print(recordlinkage.recall(true_links, df))
    print(recordlinkage.fscore(true_links, df))


def main():
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_rows', None)

    df_train = load_febrl2(return_links=True)[0]  # DataFrame to train
    df_test = load_febrl1(return_links=True)[0]  # DataFrame to test
    y_train = load_febrl2(return_links=True)[1]  # True links to train
    y_test = load_febrl1(return_links=True)[1]  # True links to test

    indexer = recordlinkage.Index().full()  # Initiate indexer
    candidate_links = indexer.index(df_train)  # Index pairs of records for the training set
    candidate_links_test = indexer.index(df_test)  # Index pairs of records for the testing set

    comp = recordlinkage.Compare()  # Initiate the compare module

    # Set up parameters and thresholds for the comparison process
    comp.string(
        "given_name",
        "given_name",
        method="jarowinkler",
        label="given_name",
        threshold=0.85
    )
    comp.string(
        "surname",
        "surname",
        method="jarowinkler",
        label="surname",
        threshold=.85
    )
    comp.string("street_number", "street_number", label="street_number", threshold=0.85)
    comp.string("address_1", "address_1", label="address_1", threshold=0.85)
    comp.string("address_2", "address_2", label="address_2", threshold=0.85)
    comp.string("suburb", "suburb", label="suburb", threshold=0.85)
    comp.string("state", "state", label="state", threshold=0.85)
    comp.string("postcode", "postcode", label="postcode", threshold=0.85)
    comp.string("date_of_birth", "date_of_birth", label="date_of_birth", threshold=0.85)
    comp.exact("soc_sec_id", "soc_sec_id", label="soc_sec_id")

    # Start comparing
    features = comp.compute(candidate_links, df_train)
    features_test = comp.compute(candidate_links_test, df_test)

    # Compute the total score from scores
    # Sort them in descending order
    print(features.sum(axis=1).value_counts().sort_index(ascending=False))

    # Classification
    matches = features[features.sum(axis=1) > 5]  # Only take the ones with total score above 5
    matches["score"] = matches.loc[:, "given_name": "soc_sec_id"].sum(axis=1)  # Add in a score column
    matches = matches.sort_values(by=["score"], ascending=False)  # Sort in descending order based on the score
    # matches.to_csv("matches.csv")  # Export data to csv file

    print(matches)

    display_quality(y_train, matches)

    # Check algorithm's matching vs. true links
    golden_pairs = features
    golden_matches_index = golden_pairs.index.intersection(y_train)

    log_reg = recordlinkage.LogisticRegressionClassifier()  # Initiate logistic regression classifier

    log_reg.fit(golden_pairs, golden_matches_index)  # Train data
    log_reg_predict = log_reg.predict(features)  # Predict data

    display_quality(y_train, log_reg_predict)

    log_reg_predict_test = log_reg.predict(features_test)  # Predict based on the trained data

    display_quality(y_test, log_reg_predict_test)


if __name__ == "__main__":
    main()
