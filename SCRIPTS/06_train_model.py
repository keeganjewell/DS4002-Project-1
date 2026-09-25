"""
PIPELINE STEP 6 of 6 (final step): train, evaluate, and save the model.

Genre classification from lyrics: TF-IDF + multinomial logistic regression.

Must run AFTER 04_collect_dataset.py and 05_backfill_country.py, since it
reads DATA/music_genre_dataset_final.csv (the balanced, 400-song dataset
those two scripts produce).

Implements the Methodology/Evaluation plan:
  - Preprocess lyrics (lowercase, tokenize, strip punctuation, remove
    stopwords). No missing-value handling needed -- rows with missing or
    non-English lyrics were already excluded during data collection.
  - TF-IDF features, filtering very rare (min_df) and very common (max_df)
    words.
  - Split by ARTIST, not song, so no artist appears in both train and test.
  - Tune with GroupKFold cross-validation (grouped by artist) on the
    training artists only.
  - Fit multinomial logistic regression on TF-IDF features.
  - Evaluate on the held-out artists' test set: accuracy, macro F1,
    confusion matrix.
  - Inspect top TF-IDF-weighted words per genre from the fitted model.
  - Refit on the FULL labeled dataset (all 400 songs) and save that model
    to disk -- this is the model the SITE/app.py website actually loads
    and serves predictions from.

Run with: python SCRIPTS/06_train_model.py
Requires: DATA/music_genre_dataset_final.csv (produced by steps 04-05);
          no internet access needed, runs in well under a minute.
Output:
  OUTPUT/model_metrics.txt        -- accuracy, macro F1, classification report
  OUTPUT/confusion_matrix.png     -- confusion matrix heatmap
  OUTPUT/top_words_per_genre.png  -- top model coefficients per genre
  OUTPUT/top_words_per_genre.csv  -- same, as a table
  MODEL/vectorizer.joblib         -- fitted TF-IDF vectorizer (full dataset)
  MODEL/classifier.joblib         -- fitted logistic regression model (full dataset)
"""

import os
import re
import string

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

RANDOM_SEED = 4002

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(BASE_DIR, "DATA")
OUTPUT_DIR = os.path.join(BASE_DIR, "OUTPUT")
MODEL_DIR = os.path.join(BASE_DIR, "MODEL")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

FINAL_PATH = os.path.join(DATA_DIR, "music_genre_dataset_final.csv")

GENRES = ["Country", "Pop", "Rock", "Hip-Hop/Rap"]
TARGET_ACCURACY = 0.75


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------

PUNCT_RE = re.compile(f"[{re.escape(string.punctuation)}]")
WHITESPACE_RE = re.compile(r"\s+")


def clean_lyrics(text):
    """Lowercase, strip punctuation, collapse whitespace. Filler-word
    removal is handled by the TfidfVectorizer's stop_words list so it stays
    in one place with the min_df/max_df filtering."""
    text = str(text).lower()                    # case-fold so "Love" and "love" count as the same token
    text = PUNCT_RE.sub(" ", text)               # strip punctuation (commas, apostrophes, etc.) to plain words
    text = WHITESPACE_RE.sub(" ", text).strip()  # collapse any resulting double-spaces/newlines
    return text


def load_data():
    # Read the final, class-balanced dataset produced by steps 04-05.
    df = pd.read_csv(FINAL_PATH)
    # Defensive drop only -- by design, rows with missing lyrics/genre/artist
    # were already excluded during collection, so this should be a no-op.
    df = df.dropna(subset=["lyrics", "target_genre", "artist"]).copy()
    df["clean_lyrics"] = df["lyrics"].apply(clean_lyrics)
    # Guard against any song whose lyrics became empty after cleaning
    # (e.g. lyrics that were purely punctuation/whitespace).
    df = df[df["clean_lyrics"].str.len() > 0].copy()
    return df


# --------------------------------------------------------------------------
# Artist-grouped train/test split
# --------------------------------------------------------------------------

def artist_grouped_split(df, test_size=0.20, random_state=RANDOM_SEED):
    # GroupShuffleSplit keeps every song from a given artist on the same
    # side of the split, so the model is tested on artists it has never
    # seen lyrics from -- a much harder (and more honest) test than a
    # random song-level split would be.
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, groups=df["artist"]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    # Hard check that no artist leaked across the split -- if this ever
    # fails, the evaluation numbers below would be misleadingly optimistic.
    train_artists = set(train_df["artist"])
    test_artists = set(test_df["artist"])
    assert train_artists.isdisjoint(test_artists), "Artist leakage between train/test!"

    return train_df, test_df


# --------------------------------------------------------------------------
# Model pipeline: TF-IDF + multinomial logistic regression, tuned with
# GroupKFold CV grouped by artist
# --------------------------------------------------------------------------

def make_vectorizer():
    """Build a fresh (unfitted) TF-IDF vectorizer with the same settings
    everywhere it's used, so train/CV/deployment all featurize text
    identically."""
    return TfidfVectorizer(
        stop_words=list(ENGLISH_STOP_WORDS),
        min_df=3,       # drop very rare words (appear in < 3 songs) -- likely noise/typos
        max_df=0.85,    # drop very common words (appear in > 85% of songs) -- too generic to be informative
        ngram_range=(1, 1),  # single words (unigrams) only, per the Methodology plan
        sublinear_tf=True,   # dampen the effect of a word repeated many times in one song (e.g. a chorus)
    )


def make_classifier(C=1.0):
    """Build a fresh (unfitted) multinomial logistic regression classifier.
    C is the inverse regularization strength -- smaller C = more
    regularization; it's tuned via cross-validation in build_and_tune_model."""
    return LogisticRegression(
        C=C,
        solver="lbfgs",   # supports multinomial (softmax) loss for >2 classes
        max_iter=2000,    # TF-IDF features need more iterations than lbfgs's default to converge
        random_state=RANDOM_SEED,
    )


def build_and_tune_model(train_df):
    """Fit TF-IDF on the training lyrics, then use GroupKFold cross-
    validation (grouped by artist, so no artist's songs span a CV
    train/validation split either) to pick the regularization strength C
    that gives the best macro F1."""
    X_text = train_df["clean_lyrics"]
    y = train_df["target_genre"]
    groups = train_df["artist"]

    vectorizer = make_vectorizer()
    param_grid = {"C": [0.1, 0.3, 1, 3, 10]}  # candidate regularization strengths to search over
    group_kfold = GroupKFold(n_splits=5)

    # Fit TF-IDF once on the full training set (not refit per fold) --
    # scikit-learn's GridSearchCV only refits the classifier per fold,
    # which matches how the plan describes tuning the model, not the
    # vectorizer, via cross-validation.
    X_tfidf = vectorizer.fit_transform(X_text)

    grid = GridSearchCV(
        make_classifier(),
        param_grid,
        cv=group_kfold.split(X_tfidf, y, groups=groups),  # pre-computed artist-grouped folds
        scoring="f1_macro",  # optimize for macro F1, not accuracy, so the smallest genre isn't ignored
        n_jobs=-1,            # use all available CPU cores to search the grid in parallel
    )
    grid.fit(X_tfidf, y)

    print(f"Best C from GroupKFold CV (macro F1): {grid.best_params_['C']}")
    print(f"Best CV macro F1: {grid.best_score_:.3f}")

    best_clf = grid.best_estimator_
    return vectorizer, best_clf, grid


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def evaluate(vectorizer, clf, test_df):
    """Score the fitted model on the held-out artists (never seen during
    training or CV tuning) and write accuracy/macro-F1/confusion-matrix
    results to OUTPUT/."""
    # Reuse the vectorizer's already-learned vocabulary/IDF weights --
    # transform, not fit_transform, so test-set words can't leak into
    # the vectorizer's fitted statistics.
    X_test = vectorizer.transform(test_df["clean_lyrics"])
    y_test = test_df["target_genre"]
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    # Macro F1 averages each genre's F1 equally, so Hip-Hop/Rap's easy
    # separability can't mask poor performance on a smaller/harder genre.
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred, labels=GENRES)

    print(f"\nHeld-out test accuracy: {acc:.3f}")
    print(f"Held-out test macro F1: {macro_f1:.3f}")
    print(f"Goal (75% accuracy): {'MET' if acc >= TARGET_ACCURACY else 'NOT MET'}")
    print("\nClassification report:")
    print(report)

    # Rows = true genre, columns = predicted genre -- diagonal is correct
    # predictions; off-diagonal cells show which genres get mixed up.
    cm = confusion_matrix(y_test, y_pred, labels=GENRES)

    with open(os.path.join(OUTPUT_DIR, "model_metrics.txt"), "w") as f:
        f.write(f"Held-out test accuracy: {acc:.3f}\n")
        f.write(f"Held-out test macro F1: {macro_f1:.3f}\n")
        f.write(f"Goal (75% accuracy): {'MET' if acc >= TARGET_ACCURACY else 'NOT MET'}\n\n")
        f.write("Classification report:\n")
        f.write(report)
        f.write("\nConfusion matrix (rows=true, cols=predicted):\n")
        f.write(pd.DataFrame(cm, index=GENRES, columns=GENRES).to_string())
        f.write("\n")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=GENRES)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"Confusion Matrix (Test Accuracy = {acc:.1%})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    return acc, macro_f1, y_pred


# --------------------------------------------------------------------------
# Top words per genre (model interpretation)
# --------------------------------------------------------------------------

def top_words_per_genre(vectorizer, clf, top_n=15):
    """For each genre, find the top_n TF-IDF words with the largest positive
    logistic regression coefficient for that genre's one-vs-rest decision --
    i.e. the words that push the model most strongly toward predicting that
    genre. Saves both a bar-chart figure and a CSV table."""
    feature_names = np.array(vectorizer.get_feature_names_out())
    classes = clf.classes_

    rows = []
    fig, axes = plt.subplots(1, len(classes), figsize=(5 * len(classes), 6), sharey=False)
    if len(classes) == 1:
        axes = [axes]

    for ax, genre in zip(axes, classes):
        class_idx = list(classes).index(genre)
        coefs = clf.coef_[class_idx]                    # one coefficient per TF-IDF vocabulary word, for this genre
        top_idx = np.argsort(coefs)[::-1][:top_n]        # indices of the top_n largest (most positive) coefficients
        top_words = feature_names[top_idx]
        top_coefs = coefs[top_idx]

        for w, c in zip(top_words, top_coefs):
            rows.append({"genre": genre, "word": w, "coefficient": c})

        # Reverse order so the single largest-coefficient word plots at the top of the bar chart.
        ax.barh(top_words[::-1], top_coefs[::-1], color="#4C72B0")
        ax.set_title(genre)
        ax.set_xlabel("Logistic regression coefficient")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "top_words_per_genre.png"), dpi=150)
    plt.close(fig)

    words_df = pd.DataFrame(rows)
    words_df.to_csv(os.path.join(OUTPUT_DIR, "top_words_per_genre.csv"), index=False)
    return words_df


# --------------------------------------------------------------------------
# Deployment model: refit on ALL labeled data (train + test) using the
# C chosen by GroupKFold CV, so the live website uses every labeled example
# instead of holding 20% out. The held-out test split above is only for
# honest accuracy/F1 reporting, not for what actually gets served.
# --------------------------------------------------------------------------

def save_deployment_model(df, best_C):
    """Refit the vectorizer + classifier on every labeled song (train and
    test combined) using the C value that GroupKFold CV already selected,
    then persist both to disk with joblib so SITE/app.py can load them
    without retraining at request time."""
    vectorizer = make_vectorizer()
    X_tfidf = vectorizer.fit_transform(df["clean_lyrics"])
    clf = make_classifier(C=best_C)
    clf.fit(X_tfidf, df["target_genre"])

    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "vectorizer.joblib"))
    joblib.dump(clf, os.path.join(MODEL_DIR, "classifier.joblib"))
    print(f"\nDeployment model (fit on all {len(df)} songs, C={best_C}) saved to {MODEL_DIR}/")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    # Step 1: load and clean the final labeled dataset.
    df = load_data()
    print(f"Loaded {len(df)} songs, {df['artist'].nunique()} artists")
    print(df["target_genre"].value_counts())

    # Step 2: split by artist so evaluation reflects generalization to
    # never-before-seen artists, not just never-before-seen songs.
    train_df, test_df = artist_grouped_split(df)
    print(f"\nTrain: {len(train_df)} songs, {train_df['artist'].nunique()} artists")
    print(f"Test:  {len(test_df)} songs, {test_df['artist'].nunique()} artists")
    print("\nTrain genre counts:")
    print(train_df["target_genre"].value_counts())
    print("\nTest genre counts:")
    print(test_df["target_genre"].value_counts())

    # Step 3: fit TF-IDF + logistic regression on the training artists,
    # tuning regularization strength C via artist-grouped cross-validation.
    vectorizer, clf, grid = build_and_tune_model(train_df)

    # Step 4: honestly evaluate on the held-out test artists.
    acc, macro_f1, y_pred = evaluate(vectorizer, clf, test_df)

    # Step 5: inspect which words drive each genre's predictions.
    words_df = top_words_per_genre(vectorizer, clf)
    print("\nTop words per genre saved to OUTPUT/top_words_per_genre.csv")

    # Step 6: refit on ALL labeled data and save the model the website serves.
    save_deployment_model(df, best_C=grid.best_params_["C"])

    print(f"\nAll outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
