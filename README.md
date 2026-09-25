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
