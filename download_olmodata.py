import os
import subprocess
import urllib.parse
import yaml
from tqdm import tqdm
import multiprocessing

def download_files_batch(urls, dest_folder):
    downloaded_files = []
    for url in urls:
        # Parse the URL to get the filename
        parsed_url = urllib.parse.urlparse(url)
        filename = os.path.basename(parsed_url.path)
        local_filepath = os.path.join(dest_folder, filename)
        
        # Check if the file already exists
        if os.path.exists(local_filepath):
            print(f"File already exists: {local_filepath}. Skipping download.")
        else:
            # Download the file using curl without progress bar
            subprocess.run(['curl', '-o', local_filepath, url], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        
        downloaded_files.append(local_filepath)
    
    return downloaded_files

def update_yaml_with_local_paths(data, dest_folder, batch_size=1):
    urls = data['data']['paths'][:20]

    urls = ['https://olmo-data.org/preprocessed/olmo-mix/v1_5-sample/gpt-neox-20b-pii-special/part-018-00000.npy',
    'https://olmo-data.org/preprocessed/olmo-mix/v1_5-sample/gpt-neox-20b-pii-special/part-018-00001.npy'
    'https://olmo-data.org/preprocessed/olmo-mix/v1_5-sample/gpt-neox-20b-pii-special/part-100-00001.npy',
    'https://olmo-data.org/preprocessed/olmo-mix/v1_5-sample/gpt-neox-20b-pii-special/part-100-00002.npy',
    'https://olmo-data.org/preprocessed/olmo-mix/v1_5-sample/gpt-neox-20b-pii-special/part-187-00001.npy',
    'https://olmo-data.org/preprocessed/olmo-mix/v1_5-sample/gpt-neox-20b-pii-special/part-187-00002.npy']

    batches = [urls[i:i + batch_size] for i in range(0, len(urls), batch_size)]
    
    new_paths = []
    
    with multiprocessing.Pool() as pool:  # Using multiprocessing Pool
        # Use multiprocessing to download batches of files in parallel
        results = pool.starmap(download_files_batch, [(batch, dest_folder) for batch in batches])

        # Flatten the list of results and update new_paths
        for result in tqdm(results, total=len(results), desc="Downloading files", unit="batch"):
            new_paths.extend(result)

    # Replace the URLs with local paths
    data['data']['paths'] = new_paths



def main():
    # Load the YAML configuration (replace 'config.yaml' with your actual file)
    train_config_path = "./OLMo/configs/official/OLMo-7B.yaml"  
    with open(train_config_path, 'r') as file:
        data = yaml.safe_load(file)

    # Specify the destination folder for downloads
    dest_folder = 'downloaded_files_olmo_100k'
    os.makedirs(dest_folder, exist_ok=True)

    # Update the YAML with local paths
    print('downloading files now...')
    update_yaml_with_local_paths(data, dest_folder)

    # Save the updated YAML configuration
    with open('config_local.yaml', 'w') as file:
        yaml.dump(data, file, default_flow_style=False)

    print("All files have been downloaded and YAML updated.")

if __name__ == "__main__":
    main()
