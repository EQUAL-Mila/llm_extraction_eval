import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

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

    # read all three texts files in the pile subfolder
    with open('pile_metadata/books1_cleaned.txt', 'r') as f:
        books1 = f.read().splitlines()
    with open('pile_metadata/books3_cleaned.txt', 'r') as f:
        books3 = f.read().splitlines()

    with open('pile_metadata/pg19.txt', 'r') as f:
        pg19 = f.read().splitlines()

    big_pile = books1 + books3 + pg19
    print("the length of the big pile is: ")
    print(len(big_pile))

    # read the pg_catalog.csv file
    pg_catalog = load_pg_metadata()

    #identify books that are not in the pile buit are in the pg_catalog
    books_outside_pile = []
    for book in tqdm(pg_catalog):
        book_names = book.lower().split()
        number_of_pieces = len(book_names)
        detected_count = 0

        for sentence in big_pile:
            if book in sentence:
                detected_count += 1
                break
            # if detected_count / number_of_pieces > 0.75:
            #     continue

        if detected_count == 0:
            books_outside_pile.append(book)
            # save the books outside the pile to a text file
            with open('pile_metadata/books_outside_pile.txt', 'w') as f:
                for item in books_outside_pile:
                    f.write("%s\n" % item)

    print("\n -- Some stats -- \n")
    print("Books outside pile:", len(books_outside_pile))
    print("Total books in pile:", len(big_pile))

if __name__ == '__main__':
    # preprocess_books('books3')
    # load_pg19_metadata()
    get_books_outside_pile()
