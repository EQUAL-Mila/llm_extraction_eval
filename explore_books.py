import json

books_outside_pile_path = "the_books_[local_7pm]_outside_pile.txt"  # Replace with the actual file path
strings_list = []
import random

def load_books_outside_pile():
    """
    Load the books outside pile from the given file path and return the list of strings
    args:
        file_path: str: The path to the file containing the books outside pile
    return:
        list: The list of strings containing the books outside pile

    """
    with open(books_outside_pile_path, "r") as file:
        for line in file:
            line = line.strip()  # Remove leading/trailing whitespace
            if line:  # Skip empty lines
                strings_list.append(line)

    print(strings_list[:10])


# function to load a book from pg19
def load_pg19_metadata():

    data = pd.read_csv('./pg19_nineteen_metadata.csv', header=None)

    print("Number of columns:", data.shape[1])

    # Print the column headers
    print("Column headers:", data.columns)

    second_column = data.iloc[:,
                              1]  # ':' means all rows, '1' is the index of the second column

    data = second_column.tolist()
    data = [i.lower() for i in data]

    # randomly pick some book from the pg-19 data
    selected_book_title = random.choice(data)

    return selected_book_title




# write a function to load all the json files in a given directory ( the books outside pile )
def load_json_downloaded():
    """
    Parse the json-metadata with respect to each book downloaded, and extract the relevant titles to match it back to the strings we have
    args:
    """
    

def load_levenshtein_ratios():
    try:
        with open('./ratios_of_books_outside.json', 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")


def get_books_according_to_levenshtein():

    ratios = load_levenshtein_ratios()
    # load_books_outside_pile()
    # sort a dictionary in descending order based on values
    sorted_ratios_descending = dict(sorted(ratios.items(), key=lambda item: item[1], reverse=True))
    sorted_ratios_ascending = dict(sorted(ratios.items(), key=lambda item: item[1], reverse=False))
    print(len(sorted_ratios_ascending))
    for item in list(sorted_ratios_ascending.items())[9950:10000]:
        print(item)





if __name__ == "__main__":

