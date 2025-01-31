# good-day-quantitative

Data management and analysis code for the paper "Exploring 'good days' with advanced cancer: A pilot daily diary study."

[Citation](https://journals.sagepub.com/doi/10.1177/02692163241310683): 

> Lazris D, Fedor J, Cheng S, et al. Exploring “good days” with advanced cancer: A pilot daily diary study. *Palliative Medicine.* 2025;39(2):318-323. doi:10.1177/02692163241310683

<br>

## Installation 

1. Install [Miniconda](https://docs.anaconda.com/free/miniconda/miniconda-install/)

2. Install [Quarto](https://quarto.org/docs/get-started/) 

3. Clone the repository:

    ```bash
    git clone https://github.com/carissalow/good-day-quantitative
    ```

4. Restore the Python virtual environment:

    ```bash
    cd good-day-quantitative
    conda env create -f environment.yml -n good-day
    ```

<br>

---

## Configuration 

Copy CSV files containing the following input data to `data/raw/`:

- Daily diary questionnaire data exported from Qualtrics  
- Participant baseline demographics exported from REDCap  
- Participant baseline PROMIS scores exported from REDCap   
- Participant study start and end dates  


**Optional:** Modify any data cleaning or analysis parameters exposed in `config.yaml`. 

<br>

---

## Execution 

Run all steps of the analysis:

```bash
bash -l run_analysis.sh
```

Steps of the analysis include:  

1. Activating the Python environment  
2. Cleaning daily diary, demographics, and PROMIS score data  
3. Computing descriptive statistics and correlations, performing t-tests, and fitting linear mixed effects regression models  
4. Creating a table and figure for our publication   
5. Rendering a summary report   

After running the analysis pipeline:  

- Cleaned data will be available in `data/processed/`
- A log of the steps performed to clean the daily diary data can be found in `logs/clean_diary_data.log`   
- Analysis results will be available in `output/descriptives/` and `output/results/`  
- The table and figure for our publication will be available in `output/tables/` and `output/figures/`, respectively   
- The rendered summary report can be found in `reports/_site/good_day_quantitative_analysis_summary.html`  

You can view our report [here](https://carissalow.github.io/good-day-quantitative).  

<br>