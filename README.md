# Client Management System

This project utilizes the RecordLinkage library to identify and classify duplicate records within a database. This information is then used to train a predictive model that helps forecast results for new data based on a model that has learned from labeled training. The model's primary function is to assess whether an individual is a potential client based on their unique profile and whether or not their personal data is already in the database. If not, generate a key to link it to that client and store them in the system's database.

I find it easier using notebooks such as Kaggle so that we can see the outputs and the dataframes without having to rerun the entire program.

## Concepts Involved

* Golden Records
* Indexing - Ex: Given 5 different clients, how many ways can we pair them together?
* Supervised & Unsupervised Learning
* Confusion Matrix
* F-Score
* Overfit

## Acknowledgements

- [Python Record Linkage](https://recordlinkage.readthedocs.io/en/latest/index.html)
- [Fuzzy Matching &amp; Record Linking on Hospital Datase](https://www.kaggle.com/code/visionary20/fuzzy-matching-record-linking-on-hospital-datase)
