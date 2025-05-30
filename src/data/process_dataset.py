from tqdm.auto import tqdm
from src.data.feature_engineering import reformat_dataset, generate_feature
from src.model_selection import process_cv_folds

def process_subject(subject, n_jobs = 1):

    eeg, category, exemplar = reformat_dataset(subject)
    
    pseudocategory = generate_feature(subject, 'pseudocategory', category=category, exemplar=exemplar)
    generate_feature(subject, 'AEP', n_jobs=n_jobs, eeg=eeg)
    generate_feature(subject, 'ST', n_jobs=n_jobs, eeg=eeg)
    generate_feature(subject, 'WPLI', n_jobs=n_jobs, eeg=eeg)

    process_cv_folds(subject, 'category', exemplar=exemplar, category=category)
    process_cv_folds(subject, 'pseudocategory', exemplar=exemplar, category=pseudocategory, pseudocategory=True)

def process_dataset(subjects, n_jobs = 1):
    for subject in tqdm(subjects, desc='Preprocessing data'):
        process_subject(subject, n_jobs)
