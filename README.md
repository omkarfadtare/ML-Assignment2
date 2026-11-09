# Dry Bean classification

## Problem statement

Predict the variety of a dry bean from 16 measurements of its shape. Compare Logistic Regression, Decision Tree, KNN, Naive Bayes, and Support Vector Machine on the same held-out test set. The app can also evaluate an uploaded labelled CSV.

## Dataset description

Source: [UCI Dry Bean dataset](https://archive.ics.uci.edu/dataset/602/dry+bean) ([DOI 10.24432/C50S4B](https://doi.org/10.24432/C50S4B)). The dataset has 13,611 records, 16 numeric features, and seven bean classes. `Class` is the target column. No feature values are missing.

We removed 68 exact duplicate HOROZ rows before splitting the data. This left 13,543 records, including 2,709 records in the supplied `test_data.csv`.

| Class | Original rows | After deduplication | Test rows |
| --- | ---: | ---: | ---: |
| BARBUNYA | 1,322 | 1,322 | 265 |
| BOMBAY | 522 | 522 | 104 |
| CALI | 1,630 | 1,630 | 326 |
| DERMASON | 3,546 | 3,546 | 709 |
| HOROZ | 1,928 | 1,860 | 372 |
| SEKER | 2,027 | 2,027 | 406 |
| SIRA | 2,636 | 2,636 | 527 |
| **Total** | **13,611** | **13,543** | **2,709** |

The features are `Area`, `Perimeter`, `MajorAxisLength`, `MinorAxisLength`, `AspectRation`, `Eccentricity`, `ConvexArea`, `EquivDiameter`, `Extent`, `Solidity`, `roundness`, `Compactness`, `ShapeFactor1`, `ShapeFactor2`, `ShapeFactor3`, and `ShapeFactor4`. The names `AspectRation` and `roundness` follow the source data's spelling and case.

The classes are uneven: DERMASON has 3,546 records, while BOMBAY has 522.

## GitHub Repository Link

[https://github.com/omkarfadtare/ML-Assignment2](https://github.com/omkarfadtare/ML-Assignment2)

## Models used

Scores shown in the Streamlit app for the supplied 2,709-record test set:

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.9192 | 0.9947 | 0.9307 | 0.9300 | 0.9302 | 0.9023 |
| Decision Tree | 0.9037 | 0.9693 | 0.9191 | 0.9176 | 0.9182 | 0.8835 |
| KNN | 0.9155 | 0.9899 | 0.9302 | 0.9265 | 0.9279 | 0.8979 |
| Naive Bayes | 0.8970 | 0.9911 | 0.9081 | 0.9082 | 0.9072 | 0.8762 |
| Support Vector Machine | 0.9210 | 0.9948 | 0.9345 | 0.9318 | 0.9330 | 0.9045 |

## Observations about model performance

| Model | Observation |
| --- | --- |
| Logistic Regression | A strong result, with 219 errors on the test set, only five more than SVM. |
| Decision Tree | Its AUC is the lowest of the five models. |
| KNN | Its scores are close to Logistic Regression, though slightly lower. |
| Naive Bayes | It has the lowest accuracy and F1, even though its AUC is high. |
| Support Vector Machine | It has the highest accuracy, F1, and MCC. It correctly classifies 2,495 of 2,709 test records. |
| Overall | SVM leads, but its accuracy is only 0.18 percentage points above Logistic Regression. |

SVM most often confuses DERMASON and SIRA: 50 DERMASON records are labelled SIRA, and 48 SIRA records are labelled DERMASON. These results come from one held-out test set, so scores may differ on new data.

