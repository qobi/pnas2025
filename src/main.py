from src.data import download_raw_dataset, process_dataset
from src.experiments.run_experiment import run
from src.analysis import analyze_results

SUBJECTS = [f'S{subject + 1}' for subject in range(10)]

def main(experiments, devices = ['cpu'], n_jobs = 1):

    download_raw_dataset(SUBJECTS)
    process_dataset(SUBJECTS, n_jobs = n_jobs)
    
    results = run(experiments, devices)
    analyze_results(**results)

if __name__ == '__main__':
    
    experiments = [
                    'paired_category_decoding', 
                    'paired_pseudocategory_decoding',
                   ]

    devices = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cuda:4', 'cuda:5', 'cuda:6', 'cuda:7']
    # devices = ['cuda:0']

    main(experiments, devices, n_jobs = -1)
    
