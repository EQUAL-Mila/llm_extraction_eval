import pandas as pd
from datasets import load_dataset


def preprocess_books(dataset='books1'):
    '''
    Preprocess the books3 and books1 metadata and saves them separately.
    '''
    if dataset == 'books3':
        name = 'pile_metadata/books3.txt'
    if dataset == 'books1':
        name = 'pile_metadata/books1.txt'

    cleaned_dataset = []
    with open(name, 'r') as f:
        lines = f.read().splitlines()
        # print(lines[:5])
        for line in lines[:]:
            filename = line.replace('\\', '/').split('/')[-1]
            filename = filename.split('.')[0]
            filename = filename.lower()
            cleaned_dataset.append(filename)

    with open(name.replace('.txt', '_cleaned.txt'), 'w') as f:
        for item in cleaned_dataset:
            f.write("%s\n" % item)

def load_pg_metadata():
    data = pd.read_csv('pg_catalog.csv', header=None, names=['Title'])
    data = data.tolist()
    data = [i.lower() for i in data]
    return data

def load_pg19_metadata():
    dataset = load_dataset("pg19")
    # iterate through the huggingface dataset and extract only the first columns
    metadata = []
    for i in range(len(dataset['train'])):
        l = dataset['train'][i]['short_book_title']
        l = l.lower()
        metadata.append(l)
    # save the metadata to a text file
    with open('pile_metadata/pg19.txt', 'w') as f:
        for item in metadata:
            f.write("%s\n" % item)

if __name__ == '__main__':
    # preprocess_books('books3')
    load_pg19_metadata()
