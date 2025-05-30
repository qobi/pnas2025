import os
import requests
from tqdm.auto import tqdm
from scipy.io import loadmat

def download_raw_data(subject):

  data_dir = os.path.join('data', 'SUDB', 'raw')
  os.makedirs(data_dir, exist_ok=True)

  data_path = os.path.join(data_dir, f'{subject}.mat')

  url = f"https://stacks.stanford.edu/file/druid:bq914sc3730/{subject}.mat"

  try:
    r = requests.get(url, stream=True)
    total_size = int(r.headers.get('content-length', 0))
    block_size = 8192
    with open(data_path, "wb") as fid:
      for data in tqdm(r.iter_content(block_size), total=total_size//block_size, unit='KB', desc=f'Downloading data for {subject}', leave=False):
      
        fid.write(data)

  except requests.ConnectionError:
    print("Failed to connect to the server. Please check your internet connection.")
  else:
    if r.status_code != requests.codes.ok:
      raise requests.HTTPError(f"Failed to download {subject}.mat: {r.status_code} {r.reason}")

def download_raw_dataset(subjects):
  for subject in subjects:
    try:
      load_raw_data(subject)
    except FileNotFoundError:
      download_raw_data(subject)

def load_raw_data(subject):
  data_path = os.path.join('data', 'SUDB', 'raw', f'{subject}.mat')
  return loadmat(data_path)