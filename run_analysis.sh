#!/bin/bash
conda activate good-day

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running data cleaning scripts..."
python src/clean_diary_data.py
python src/clean_demographics_data.py
python src/clean_baseline_promis_data.py

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running analysis scripts..."
python src/compute_descriptive_statistics.py
python src/compute_icc.py
python src/compute_between_person_correlations.py
python src/fit_univariable_linear_mixed_models.py
python src/create_model_diagnostic_plots.py
python src/compute_within_person_correlations.py

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Creating figures and tables..."
python src/create_figure_1.py
python src/create_table_1.py

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Rendering Quarto report..."
quarto render reports

conda deactivate
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done!"