# Part-of-Speech Tagging

This project implements two different approaches for part-of-speech tagging on the Brown Corpus:

- Hidden Markov Model (HMM)
- Logistic Regression (LR)

Both models use the Universal Tagset provided by NLTK.

## Files

- `hmm_template.py`: HMM-based POS tagging implementation
- `lr_template.py`: Logistic Regression-based POS tagging implementation
- `hmm_output.txt`: Reference output for the HMM part
- `docs/PA3.pdf`: Assignment description

## Requirements

- Python 3
- `nltk`
- `numpy`
- `scikit-learn`

You can install the required packages with:

```powershell
python -m pip install nltk numpy scikit-learn
```

## Dataset

The project uses the Brown Corpus from NLTK together with the Universal Tagset mapping.

If the dataset is not already available, download it with:

```powershell
python -m nltk.downloader brown universal_tagset
```

## Running the project

Run the HMM model:

```powershell
python hmm_template.py
```

Run the Logistic Regression model:

```powershell
python lr_template.py
```
