import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from Levenshtein import ratio
import numpy as np
from FlagEmbedding import FlagModel
import json
import pickle


def dump_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)
    

def save_books(set_of_books, filename):
    # save the books as a file
    with open(filename, 'w') as f:
        for item in set_of_books:
            f.write("%s\n" % item)

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
    # complete gutenberg catalog/metadata

    data = pd.read_csv('pg_catalog.csv')
    data = data['Title']
    data = data.tolist()
    data = [i.lower() for i in data]
    return data


def load_pg19_metadata():

    data = pd.read_csv('./pg19_nineteen_metadata.csv', header=None)

    print("Number of columns:", data.shape[1])

    # Print the column headers
    print("Column headers:", data.columns)

    second_column = data.iloc[:,
                              1]  # ':' means all rows, '1' is the index of the second column

    data = second_column.tolist()
    data = [i.lower() for i in data]

    # save the metadata to a text file
    with open('pile_metadata/pg19.txt', 'w') as f:
        for item in data:
            f.write("%s\n" % item)


def get_books_outside_pile():

    ratios_of_books_outside = {}
    # read all three texts files in the pile subfolder
    with open('pile_metadata/books1_cleaned.txt', 'r') as f:
        books1 = f.read().splitlines()
        books1 = [i.replace('-', ' ') for i in books1]
        print("printing preview")
        print(books1[:2])

    with open('pile_metadata/books3_cleaned.txt', 'r') as f:
        books3 = f.read().splitlines()
        books3 = [i.replace('-', ' ') for i in books3]
        books3 = [i.lower() for i in books3]
        print("printing preview")
        print(books3[:2])

    with open('pile_metadata/pg19.txt', 'r') as f:
        pg19 = f.read().splitlines()

    big_pile = books1 + books3 + pg19

    print("the length of books1 is: ", len(books1))
    print("the length of books3 is: ", len(books3))
    print("the length of pg19 is: ", len(pg19))
    print("the length of the big pile is: ")
    print(len(big_pile))
    print("\n ---- \n")
    # read the pg_catalog.csv file
    pg_catalog = load_pg_metadata()
    print("the length of the pg metadata is: ", len(pg_catalog))

    #identify books that are not in the pile buit are in the pg_catalog
    books_outside_pile = set()

    books_outside_pg19, books_outside_books1, books_outside_books3 = set(
    ), set(), set()

    stats = {
        'in_books1': 0,
        'in_books3': 0,
        'in_pg19': 0,
    }

    # !BOOKS 1

    # model = SentenceTransformer('llmrails/ember-v1')

    for book in tqdm(pg_catalog):
        book_names = book.lower().split()
        number_of_pieces = len(book_names)
        detected_count = 0
        max_ratio_val = 0

        for sentence in books1:
            ratio_val = ratio(sentence.split(), book.split())
            max_ratio_val = max(max_ratio_val, ratio_val)

            if ratio_val > 0.70:
                detected_count += 1
                stats['in_books1'] += 1
                break

        if detected_count == 0:

            books_outside_pile.add(book)
            books_outside_books1.add(book)
            detected_count = 0

            for sentence in books3:
                ratio_val = ratio(sentence.split(), book.split())
                max_ratio_val = max(max_ratio_val, ratio_val)
                if ratio_val > 0.70:
                    detected_count += 1
                    stats['in_books3'] += 1
                    break

        if detected_count == 0:

            books_outside_pile.add(book)
            books_outside_books3.add(book)
            detected_count = 0

            for sentence in pg19:
                ratio_val = ratio(sentence.split(), book.split())
                max_ratio_val = max(max_ratio_val, ratio_val)

                if ratio_val > 0.70:
                    detected_count += 1
                    stats['in_pg19'] += 1
                    break

        if detected_count == 0:
            books_outside_pile.add(book)
            books_outside_pg19.add(book)

        ratios_of_books_outside[book] = max_ratio_val

    print("Books outside books1:", len(books_outside_books1))
    print("Books outside books3:", len(books_outside_books3))
    print("Books outside pg19:", len(books_outside_pg19))

    print("Books outside pile:", len(books_outside_pile))
    print("Total books in pile:", len(big_pile))

    print(stats)

    dump_json(ratios_of_books_outside, 'ratios_of_books_outside.json')

    save_books(books_outside_pile, 'the_books_[cluster_7pm]_outside_pile.txt')

if __name__ == '__main__':
    # preprocess_books('books3')
    # load_pg19_metadata()
    get_books_outside_pile()
