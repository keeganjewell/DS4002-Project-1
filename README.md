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
│   └── temp
│
├── SCRIPTS/
│   ├── 01_MusicBrainz_pilot.ipynb
│   ├── 02_Lyric_collection.ipynb
│   ├── collect_dataset.py
│   ├── resume_from_checkpoint.py
│   └── scale_check.py
│
├── LICENSE
└── README.md
```

## Reproducing the Results

The repository contains both pilot code developed during the dataset-establishment stage and the final dataset collection pipeline. The final dataset can be reproduced using the following steps.

1. Clone or download this GitHub repository.

2. Ensure Python 3 is installed.

3. Install the Python packages required for dataset collection:

   ```bash
   pip install pandas requests langdetect
   ```

4. From the root directory of the repository, run:

   ```bash
   python SCRIPTS/collect_dataset.py
   ```

5. The script retrieves song and artist metadata from MusicBrainz, applies artist popularity filtering using ListenBrainz, retrieves song lyrics from LRCLIB, detects lyric language, and retains English-language songs meeting the project selection criteria.

6. The data collection pipeline generates the following files in the `DATA` directory:

   - `music_genre_dataset_raw.csv` — candidate songs selected before lyric retrieval.
   - `music_genre_dataset_with_lyrics.csv` — candidate songs after lyric retrieval but before the final language and sampling filters.
   - `music_genre_dataset_final.csv` — final English-language dataset used for project analysis.

The final dataset is designed to contain 400 songs divided evenly among four genres: Country, Pop, Rock, and Hip-Hop/Rap. Each genre contains 20 artists with 5 songs per artist.

The files `01_MusicBrainz_pilot.ipynb` and `02_Lyric_collection.ipynb` document the pilot data-collection process developed before the final pipeline. They are included for documentation but are not required to reproduce the final dataset.

`scale_check.py` was used during development to evaluate whether the proposed sampling criteria could produce the target dataset size and is not required for normal reproduction.

`resume_from_checkpoint.py` is a recovery utility that can be used if execution of `collect_dataset.py` is interrupted after a checkpoint file has been created. It is not required during a normal run.


