# Query Expansion with LLMs
This repository contains code for finetuning a T5 language model to perform query expansion using a dataset of queries and their expanded versions.

# Dataset
The dataset used for training is a collection of queries and their corresponding expanded queries. The dataset was created by collecting queries from a search engine and using a language model to generate expanded versions of these queries. The dataset is stored in a CSV file with two columns: `query` and `expanded`.

## Training
To train the model, run the following command:
```bash
python train.py
```
You will need to change the `DATASET_PATH` variable in `train.py` to point to the location of your dataset.

### Example Dataset
The dataset should be in the following format:
```csv
query,expanded
"what is the capital of France","Paris, capital of France, French capital"
"how to cook pasta","cooking pasta, how to make pasta, pasta recipe"
```