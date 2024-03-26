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
### loc is the point between the prompt p and the completion x.
df['loc'] = 1000

df.to_csv(FILENAME, index=False)