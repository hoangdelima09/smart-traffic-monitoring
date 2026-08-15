# Data

## Demo sample

- Name: *Traffic at dusk (time lapse).webm*
- Source: [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Traffic_at_dusk_(time_lapse).webm)
- Author/attribution: Editor (original YouTube channel, license reviewed by Wikimedia Commons)
- License: [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/)
- Local file: `data/samples/traffic.webm`
- Published size: 15,925,666 bytes; 1920×1080; about 23 seconds
- Published SHA-1: `918efce066535267661dcc973b79710c469298c3`
- Purpose: reproducible YOLO/ByteTrack smoke demo of cars on US Route 101 at dusk

Download with `python scripts/download_sample_data.py`. The downloaded media is not
committed by default and remains under its original license.

## Local datasets

Place manually acquired BDD100K data under `data/raw/`; generated subsets go under
`data/datasets/`. Both locations are ignored by Git except for placeholder files.
