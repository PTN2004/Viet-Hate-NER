import os
import requests


def download_dataset(url:str, save_name:str):
    response = requests.get(url)
    
    with open(save_name, "wb") as f:
        f.write(response.content)
        
    print(f"[Success] - Downlad successed {url.split("/")[-1]}")
    
    
main_url = "https://raw.githubusercontent.com/phusroyal/ViHOS/refs/heads/master/data/Sequence_labeling_based_version/Word/"
file_names = ["train_BIO_Word.csv", "dev_BIO_Word.csv", "test_BIO_Word.csv"]

for file_name in file_names:
    save_dir = "./data/"
    url = main_url + "/" + file_name
    download_dataset(url, save_dir + file_name)