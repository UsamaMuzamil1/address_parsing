
import pandas as pd
import re, math
import numpy as np
import pandas as pd
import predictions
from scipy.special import softmax

from simpletransformers.ner import NERModel
import logging

from statsmodels.tsa.base import prediction

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

columns = ['name', 'street', 'subarea', 'area',
           'city', 'province', 'Address']

count = 0
labels = set()


def generate_data(filename):
    data = pd.read_csv('corpus/dataset/extracttrain.csv')

    def get_value(x):
        global count
        results = []
        address = x["Address"]
        previous = ""
        _temp = x.to_dict()
        for each in address.split():
            match = False
            for col in columns:
                if each.strip(" ,.-") in str(_temp[col]):
                    if previous == col:
                        results.append((count, each, "I-" + col))
                        labels.add("I-" + col)
                    else:
                        results.append((count, each, "B-" + col))
                        labels.add("B-" + col)
                    previous = col
                    match = True
                    break
            if not match:
                results.append((count, each, "O"))
                labels.add("O")
                match = False
        # results.append(("", "",""))
        count += 1
        return results

    values = data[:].apply(lambda x: get_value(x), axis=1)
    flat_list = [item for sublist in values for item in sublist]
    train_df = pd.DataFrame(flat_list, columns=["sentence_id", "words", "labels"])
    return train_df


train_df = generate_data("corpus/dataset/extracttrain.csv")

test_df = generate_data("corpus/dataset/extracttest.csv")

train_df.to_csv(r'pandas.txt', header=None, index=None, sep=' ', mode='a')


arguments = {"save_eval_checkpoints": False,
             "save_steps": -1,
             'overwrite_output_dir': True,
             "save_model_every_epoch": False,
             'reprocess_input_data': True,
             "train_batch_size": 5,
             'num_train_epochs': 50,
             "max_seq_length": 256,
             "gradient_accumulation_steps": 8,
             "learning_rate": 4e-5,
             "do_lower_case": False,
             "adam_epsilon": 1e-8,
             "eval_batch_size": 8,
             "fp16": True
             }


model = NERModel(
    "bert",
    "distilbert-base-uncased",

    args=arguments,
    use_cuda= False,
    labels =  list(labels)
)
print(labels)


'''# Train the model
epochs= 5
model.train_model(train_df)


# Evaluate the model
result, model_outputs, predictions = model.eval_model(train_df)
print(result)
result, model_outputs1, predictions1 = model.eval_model(test_df)
print(result)'''



# Predictions on arbitary text strings
sentences = ["City Glass,Faizan Street,Block E,North Nazimabad,Karachi,Sindh,City Glass Faizan Street Block E North Nazimabad Karachi Sindh,"]
predictionss, raw_outputs= model.predict(sentences)
print(predictionss)














