# BiScope: AI-generated Text Detection by Checking Memorization of Preceding Tokens

Shield: [![CC BY-NC-ND 4.0][cc-by-nc-nd-shield]][cc-by-nc-nd]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-NoDerivs 4.0 International License][cc-by-nc-nd].

[![CC BY-NC-ND 4.0][cc-by-nc-nd-image]][cc-by-nc-nd]

[cc-by-nc-nd]: http://creativecommons.org/licenses/by-nc-nd/4.0/
[cc-by-nc-nd-image]: https://licensebuttons.net/l/by-nc-nd/4.0/88x31.png
[cc-by-nc-nd-shield]: https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg

---

Table of Contents
---
- [Overview](#overview)
- [Dataset](#dataset)

## Overview
- This is the official implementation for NeurIPS 2024 paper "[BiScope: AI-generated Text Detection by Checking Memorization of Preceding Tokens](https://neurips.cc/virtual/2024/poster/95814)".
- [[video](https://neurips.cc/virtual/2024/poster/95814)\] | \[[slides](https://neurips.cc/media/neurips-2024/Slides/95814.pdf)\] | \[[poster](https://neurips.cc/media/PosterPDFs/NeurIPS%202024/95814.png?t=1733630920.046255)\] | \[[paper](https://openreview.net/pdf?id=Hew2JSDycr)\]
  
<img src="Imgs/Overview.png" width="900px"/>

## Dataset
- We extend existing datasets by crafting more AI-generated data using five latest commercial LLMs, including GPT-3.5-Turbo, GPT-4-Turbo, Claude-3-Sonnet, Claude-3-Opus, and Gemini-1.0-Pro.
- Our Datasets consist of 2 short natural language datasets (Arxiv, Yelp), 2 long natural language datasets (Creative, Essay), and 1 code dataset (Code).
- We craft both the non-paraphrased version (`./Dataset`) and paraphrased version (`./Paraphrased_Dataset`) for each AI-generated data.
- Detailed dataset statistics:
<img src="Imgs/Dataset.png" width="900px"/>

## Code Implementation
Due to delays in the university's internal processing, we need to postpone the release of our code to ensure compliance with their policies. We will update and publish the code as soon as the internal procedures are completed.
