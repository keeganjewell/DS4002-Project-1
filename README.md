# Lyrical Classification of a Song's Genre

DS 4002 Project 1

Group: The Good Guys

Team Members:
- Keegan Jewell
- Nathan Nguyen
- Brian Ryu

## Contents of the Repository

## Software and Platform
### Software
- Python 3, used for data collection, preprocessing, exploratory analysis, and modeling
- Google Colab, used to develop and execute the Python notebooks
- GitHub, used to store and organize the final project repository

### Python Packages
The following Python packages are used in this project:

- pandas
- requests
- scikit-learn
- matplotlib
- langdetect

### Platform
The data collection and analysis were developed primarily in Google Colab, which runs in a Linux-based environment. Project files are stored and organized through Google Drive during development, and the final project repository is hosted on GitHub.
## Map of Documentation

The repository is organized as follows:

```text
DS4002-Project-1/
├── DATA/
│   ├── music_genre_dataset_raw.csv
│   ├── music_genre_dataset_with_lyrics.csv
│   └── music_genre_dataset_final.csv
│
├── OUTPUT/
│   ├── classification_metrics.csv
│   ├── confusion_matrix.png
│   └── top_tfidf_terms_by_genre.csv
│
├── SCRIPTS/
│   ├── 01_MusicBrainz_pilot.ipynb
│   ├── 02_Lyric_collection.ipynb
│   ├── 05_Genre_Classification.ipynb
│   ├── collect_dataset.py
│   ├── resume_from_checkpoint.py
│   └── scale_check.py
│
├── LICENSE
└── README.md
```
## Reproducing the Results

The repository contains the final dataset, the code used to construct it, and the notebook used to perform the genre-classification analysis.

### 1. Install Required Software

Python 3 is required. Install the packages used in the project with:

```bash
pip install pandas numpy requests scikit-learn matplotlib langdetect
```

### 2. Reproduce the Dataset

The final dataset is already provided in `DATA/music_genre_dataset_final.csv`.

To regenerate the dataset from the original data sources, run:

```bash
python SCRIPTS/collect_dataset.py
```

The script:

1. Retrieves candidate recordings and artist metadata from MusicBrainz.
2. Uses ListenBrainz popularity information to retain artists meeting the project's minimum listener threshold.
3. Retrieves song lyrics from LRCLIB.
4. Detects lyric language and retains English-language songs.
5. Samples 20 artists per genre and 5 songs per artist.

The resulting final dataset contains 400 songs:

- 100 Country songs
- 100 Pop songs
- 100 Rock songs
- 100 Hip-Hop/Rap songs

Each genre contains 20 artists with 5 songs per artist.

The data-collection pipeline produces:

- `DATA/music_genre_dataset_raw.csv` — candidate songs before lyric retrieval.
- `DATA/music_genre_dataset_with_lyrics.csv` — candidate songs after lyric retrieval and before final filtering.
- `DATA/music_genre_dataset_final.csv` — final balanced dataset used for modeling.

### 3. Reproduce the Classification Analysis

Open and run:

```text
SCRIPTS/05_Genre_Classification.ipynb
```

The notebook loads `music_genre_dataset_final.csv` directly from the GitHub repository and performs the following analysis:

1. Validates the final 400-song dataset.
2. Creates an artist-disjoint 80/20 train/test split:
   - 320 training songs
   - 80 test songs
   - no artist appears in both sets
3. Converts lyrics to TF-IDF features using unigrams and bigrams.
4. Trains a multinomial genre classifier using logistic regression.
5. Tunes the logistic regression regularization parameter using artist-grouped cross-validation on the training data.
6. Evaluates the final tuned model on the held-out test set using accuracy, macro F1-score, precision, recall, and a confusion matrix.
7. Identifies the TF-IDF terms most strongly associated with each genre.

The final tuned model uses `C = 10` and achieved:

- Test accuracy: 0.438
- Macro F1-score: 0.456

### 4. Generated Outputs

Running the final analysis produces the following files:

- `OUTPUT/classification_metrics.csv` — precision, recall, and F1-score results for the final tuned classifier.
- `OUTPUT/confusion_matrix.png` — confusion matrix for predictions on the held-out test set.
- `OUTPUT/top_tfidf_terms_by_genre.csv` — terms with the strongest positive logistic regression coefficients for each genre.

The pilot notebooks `01_MusicBrainz_pilot.ipynb` and `02_Lyric_collection.ipynb` document earlier dataset-development work and are not required to reproduce the final model.

`scale_check.py` was used during development to evaluate whether the sampling criteria could produce the target dataset size.

`resume_from_checkpoint.py` is a recovery utility for interrupted dataset-collection runs and is not required during a normal reproduction of the project.
