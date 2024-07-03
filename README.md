![Logo](Misc/logo-dcst-colour-1.png)

# Code Repository for the MPhil Project: 
## "Discovery and Ontology Matching of Financial Regulatory Information"

Submitted in partial fulfillment of the requirements for the Master of Philosophy in Advanced Computer Science

---

Due to size, we submit only the following trained models (given by folders with their unique IDs) with this repository:

- xb6pcE: Blocks
- 4wvN23: GraphSeg
- oiHQRN: Sentence-level model
- or6jqv: Token-base-pk-main
- z1qKSA: Token-level model trained on detailed label levels [1,2]

Other trained models can be provided on request.

## Content of this code repository:

- Data:
    - Contains the pre-trained models under `results\saved_model`
    - Exemplary PDFs that were officially published and are available online:
        - https://www.wipo.int/export/sites/www/about-wipo/en/pdf/wipo_financial_regulations.pdf
        - https://www.efsa.europa.eu/sites/default/files/corporate_publications/files/finregulation.pdf
        - https://eur-lex.europa.eu/eli/dir/2018/843/oj
        - https://www.legislation.gov.uk/uksi/2017/692/contents
- DataAnalysis:
    - Comparison tool and file_analysis.ipynb for analysis of dataset.
    - List with latitude/longitude and names of countries. Publicly available at https://developers.google.com/public-data/docs/canonical/countries_csv
    - Snippet code would contain the snippeting algorithm provided by RegGenome. Not submitted here due to confidentiality reasons.
    - Different plots as shown in the report.
- DataPreprocessing: provides different functions used as part of the data pre-processing and handling for different parts of the project.
- FRIDAY: contains main scripts for this work
    - centrol config file used to configure all scripts
    - Scripts for token-level, sentence-level, Ensemble, Blocks & GraphSeg models
    - JSON for centralised split definition, contains the IDs of documents in this works dataset
    - Script for evaluation of models in training/validation/test and on the downstream task
    - optuna_studies: contains db entries for the optuna HPO studies conducted for the token-level and sentence-level model
    - Evaluation: contains data for the evaluation chapter of the report from different models as well as evaluation code
- Misc:
    - different images from the report

## Note

For confidentiality reasons, we cannot provide:
- The raw data (i.e. the financial documents containing FRI) and thus the pre-processed data or datasets used for model training and evaluation.
- The snippeting algorithm that RegGenome provided as a baseline for this project, as outlined in Section 3.1 of the dissertation.
- Access to the API used to pull the PDF pages for the comparison tool implemented in `DataAnalysis/comparison_tool.ipynb`.

This data is only to be released with permission of the owners at Regulatory Genome Development LTD (Reg-Genome): [https://reg-genome.com/](https://reg-genome.com/)

---

**Date:** 03 June 2024
