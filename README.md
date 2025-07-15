#  Katz Cyclic Vector Classifier
A symbolic and machine learning-powered application that classifies whether a given vector is cyclic in a differential module using Katz’s algorithm and a trained ML model.
##  Project Overview
This project integrates **symbolic mathematics** and **machine learning** to:
-  Compute cyclic vectors using Katz's recursive formula.
-  Predict cyclicity from numerical matrix features using a trained Random Forest classifier.
-  Provide a Streamlit-based interactive UI for exploration and visualization.
##  Mathematical Background
A vector (v) in a differential module(V,D) is **cyclic** if the vectors ( v, Dv, D^2v,.....,D^{n-1}v ) span the entire space ( V ).  
Nicholas Katz (1987) provided a recursive construction to verify this condition using:
- A derivation matrix ( D )
- A chosen basis ( {e_0, ..........., e_{n-1}} )
- A scalar shift ( a )
##  Machine Learning Model
We extract features such as:
- Entries of matrix ( D ) and basis vectors
- Trace, determinant, rank, eigenvalues of ( D )
- Basis vector norms and rank
These are used to train a Random Forest Classifier to predict cyclicity, complementing the symbolic result.
## Directory Structure
 
 katz_cyclic_vector_ml/ 
    app/ # Streamlit web app
    data/ # Raw and processed CSV datasets
    models/ # Saved ML model + metrics
    notebooks/ # EDA and feature exploration
    outputs/ # Figures and logs
    scripts/ # Dataset generator script
    src/ # Core logic and ML pipeline
    tests/ # Unit tests
    README.md
Sample Output : Symbolic Katz derivatives displayed
Rank evaluation at fixed at t
ML predicted class : Mismatch warning (if symbolic ≠ ML result)
References
Nicholas Katz, A Simple Algorithm for Cyclic Vectors, AMS, 1987.
Andrea Pulita, Small Connections Are Cyclic, arXiv.
