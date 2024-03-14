import random
import pandas as pd

MAX_INDEX = 100000*1024
MIN_INDEX = 0

NUM_SENTENCES = 100000

FILENAME = 'demoidx100000.csv'

### Set random seed and generate a sorted list of indices
random.seed(0)
selected_indices = random.sample(range(MIN_INDEX, MAX_INDEX), NUM_SENTENCES)
selected_indices.sort()

df = pd.DataFrame({'index': selected_indices})
### Always pick the prompt from the start of the sentence (loc=0). We can make changes to this later.
df['loc'] = 0

df.to_csv(FILENAME, index=False)