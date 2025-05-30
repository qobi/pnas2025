from src.analysis import analyze_results
import pandas as pd

category_results = pd.read_csv('outputs/paired_category_decoding/results.csv')
pseudocategory_results = pd.read_csv('outputs/paired_pseudocategory_decoding/results.csv')
analyze_results(paired_category_decoding= category_results, paired_pseudocategory_decoding= pseudocategory_results)
