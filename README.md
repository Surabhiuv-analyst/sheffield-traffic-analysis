# Sheffield Road Traffic Analysis

MSc Business Analytics and Big Data — University of Liverpool
Supervisor: Dr Ehsan Amirnazmiafshar

## Project Overview
Data-driven analysis of road traffic patterns and traffic intensity
in Sheffield using DfT Raw Counts data (2000–2025).

## Data Sources
UK Department for Transport — roadtraffic.dft.gov.uk
- dft_rawcount_local_authority_id_159.csv
- dft_countpoints_local_authority_id_159.csv

## Script Execution Order
1. sheffield_analysis.py — Phase 1 and 2 EDA, generates plots 1-7
2. data_cleaning_pipeline.py — Data cleaning, feature engineering, ML models
3. composition_check.py — Vehicle composition verification
4. cyclist_check.py — Cyclist proportion verification
5. study_area_map.py — Study area map generation

## Requirements
Python 3.14
pandas, numpy, matplotlib, seaborn, scikit-learn
