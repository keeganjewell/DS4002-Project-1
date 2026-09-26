# Lyrical Classification of a Song's Genre

DS 4002 Project 1

Group: The Good Guys

Team Members:
- Keegan Jewell
- Nathan Nguyen
- Brian Ryu

## Contents of the Repository

This repository contains the data, scripts, and outputs used to collect a
balanced dataset of song lyrics and train a genre classifier that predicts
one of four genres (Country, Pop, Rock, Hip-Hop/Rap) from a song's lyrics.

## Software and Platform
### Software
- Python 3, used for data collection, preprocessing, exploratory analysis, and modeling
- Flask, used to serve the trained model through a simple web interface
- GitHub, used to store and organize the final project repository

### Python Packages
The following Python packages are used in this project:

- pandas
- numpy
- requests
- scikit-learn
- matplotlib
- langdetect
- joblib
- flask

### Platform
Data collection and modeling scripts were developed and run locally with
Python 3, using MusicBrainz, ListenBrainz, and LRCLIB as data sources over
the network. The project files are stored and organized through Google
Drive during development, and the final project repository is hosted on
GitHub.

## Map of Documentation

The repository is organized as follows:

```text
DS4002-Project-1/
├── DATA/
│   ├── README.md                          <- data summary, provenance, license,
│   │                                          ethical statements, data dictionary,
│   │                                          and explanatory plots
│   ├── music_genre_dataset_raw.csv
│   ├── music_genre_dataset_with_lyrics.csv
│   ├── music_genre_dataset_final.csv
│   ├── genre_class_balance.png            <- explanatory plot 1
│   └── lyric_length_by_genre.png          <- explanatory plot 2
│
├── MODEL/
│   ├── vectorizer.joblib
│   └── classifier.joblib
│
├── OUTPUT/
│   ├── model_metrics.txt
│   ├── confusion_matrix.png
│   ├── top_words_per_genre.csv
│   └── top_words_per_genre.png
│
├── SCRIPTS/
│   ├── 01_MusicBrainz_pilot.ipynb
│   ├── 02_Lyric_collection.ipynb
│   ├── 03_scale_check.py
│   ├── 04_collect_dataset.py
│   ├── 05_backfill_country.py
│   └── 06_train_model.py
│
├── SITE/
│   ├── app.py
│   └── templates/
│       └── index.html
│
├── LICENSE.md
└── README.md
```

## Reproducing the Results

The repository contains the final dataset, the code used to construct it,
and the script used to perform the genre-classification analysis.

### 1. Install Required Software

Python 3 is required. Install the packages used in the project with:

```bash
pip install pandas numpy requests scikit-learn matplotlib langdetect joblib flask
```

### 2. Reproduce the Dataset

The final dataset is already provided in `DATA/music_genre_dataset_final.csv`.

To regenerate the dataset from the original data sources, run:

```bash
python SCRIPTS/04_collect_dataset.py
python SCRIPTS/05_backfill_country.py
```

The pipeline:

1. Retrieves candidate recordings and artist metadata from MusicBrainz.
2. Uses ListenBrainz popularity information to retain artists meeting the project's minimum listener threshold (>=1,000 unique listeners).
3. Retrieves song lyrics from LRCLIB.
4. Detects lyric language and retains English-language songs.
5. Samples 20 artists per genre and 5 songs per artist.

`04_collect_dataset.py` on its own under-fills Country (its single
`"country"` tag search yields only ~13-15 eligible artists). This is why `05_backfill_country.py`
was created to be run right after. This will widen the Country tag search and
backfills the remaining artists needed to reach 20 artists / 100 songs for
Country, so that all four genres end up balanced.

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

See `DATA/README.md` for the full data summary, provenance, license, ethical
considerations, data dictionary, and explanatory plots for these files.

### 3. Reproduce the Classification Analysis

Run:

```bash
python SCRIPTS/06_train_model.py
```

The script loads `DATA/music_genre_dataset_final.csv` and performs the
following analysis:

1. Validates the final 400-song dataset.
2. Creates an artist-disjoint 80/20 train/test split:
   - 320 training songs
   - 80 test songs
   - no artist appears in both sets
3. Converts lyrics to TF-IDF features using unigrams, filtering very rare and very common words.
4. Trains a multinomial genre classifier using logistic regression.
5. Tunes the logistic regression regularization parameter (`C`) using artist-grouped cross-validation on the training data.
6. Evaluates the final tuned model on the held-out test set using accuracy, macro F1-score, precision, recall, and a confusion matrix.
7. Identifies the TF-IDF terms most strongly associated with each genre.
8. Refits the vectorizer and classifier on the full 400-song dataset and saves the deployment model to `MODEL/`.

The final tuned model uses `C = 10` and achieved:

- Test accuracy: 0.600
- Macro F1-score: 0.602

### 4. Generated Outputs

Running the final analysis produces the following files:

- `OUTPUT/model_metrics.txt` — accuracy, macro F1-score, classification report, and confusion matrix (as text) for the final tuned classifier.
- `OUTPUT/confusion_matrix.png` — confusion matrix heatmap for predictions on the held-out test set.
- `OUTPUT/top_words_per_genre.csv` / `OUTPUT/top_words_per_genre.png` — terms with the strongest positive logistic regression coefficients for each genre.
- `MODEL/vectorizer.joblib` / `MODEL/classifier.joblib` — the TF-IDF vectorizer and logistic regression model, refit on the full labeled dataset, used to serve live predictions.

The pilot notebooks `01_MusicBrainz_pilot.ipynb` and `02_Lyric_collection.ipynb`
document earlier dataset-development work and are not required to reproduce
the final model.

`03_scale_check.py` was used during development to check whether the
sampling criteria could realistically produce the target 400-song dataset
before committing to the full collection run in `04_collect_dataset.py`; it
is not required for a normal reproduction of the project.

### 5. Running the Website

A small Flask app in `SITE/app.py` loads the saved model from `MODEL/` and
serves a page where pasted lyrics are classified into one of the four
genres with per-genre confidence scores:

```bash
python SITE/app.py
```

Then open `http://localhost:5000`. The model files in `MODEL/` must already
exist (produced by `06_train_model.py`) before running the site.

## References

[1] MetaBrainz Foundation, "MusicBrainz Database," MusicBrainz. [Online]. Available: https://musicbrainz.org/

[2] MetaBrainz Foundation, "ListenBrainz," ListenBrainz. [Online]. Available: https://listenbrainz.org/

[3] LRCLIB, "LRCLIB API." [Online]. Available: https://lrclib.net/

[4] N. Shuyo, "langdetect: Language detection library for Python," PyPI. [Online]. Available: https://pypi.org/project/langdetect/

[5] F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825-2830, 2011.

[6] W. McKinney, "Data Structures for Statistical Computing in Python," in *Proc. 9th Python in Science Conf.*, 2010, pp. 56-61.

[7] J. D. Hunter, "Matplotlib: A 2D Graphics Environment," *Computing in Science & Engineering*, vol. 9, no. 3, pp. 90-95, 2007.

[8] Pallets Projects, "Flask," Pallets. [Online]. Available: https://flask.palletsprojects.com/
