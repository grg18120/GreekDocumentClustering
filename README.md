# Text Clustering on Modern Greek Textual Corpora

### 📄 Academic
This repository holds the code for the paper titled *"-----"*.

---

### 📁 Setup Instructions

- Before running any code, you **MUST** set the `ROOT_DIR` variable in: `GreekDocumentClustering/GrDocClust/config.py` 

- Ensure you have installed the necessary Python dependencies (e.g., scikit-learn, spaCy, etc.) from **requirements.txt**.

-  Ensure you have numpy version 1.X.X.

---

### ℹ️ Info

- `GreekDocumentClustering/Datasets/`  
  Contains the textual datasets used for clustering tasks (ZIP).

- `GreekDocumentClustering/NLP_Models_Cached_Local/`  
  Used to store NLP models locally (e.g., spaCy, transformers), preventing re-downloads and improving runtime.

- `GreekDocumentClustering/Precomputed_vectors/`  
  Stores precomputed document embeddings for each dataset and vectorization method to reduce repeated computations.

- `GreekDocumentClustering/Result/`  
  Contains CSV files with clustering evaluation metrics and results.

---

### 📝 Notes

- This project is focused on **Modern Greek** texts and aims to explore clustering techniques and evaluation metrics.
- Contributions or suggestions are welcome.

---



This repository is provided for academic use.



-----------------------------------------------------------
pip install -U pip setuptools wheel
pip install -U spacy
python -m spacy download el_core_news_lg

pip install scikit-learn
pip install scikit-learn-extra # EDW TO ERROR

pip install sentence-transformers==4.1.0

pip install einops #gia to tzina





