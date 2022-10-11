import os
import transformers
from transformers.data.processors.utils import InputFeatures
from transformers import DataProcessor, InputExample
from transformers.data.processors.glue import *
from transformers.data.metrics import glue_compute_metrics
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class TextClassificationProcessor(DataProcessor):
    def __init__(self, task_name):
        self.task_name = task_name 

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )
  
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "train.tsv"), sep='\t').values.tolist(), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "dev.tsv"), sep='\t').values.tolist(), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(pd.read_csv(os.path.join(data_dir, "test.tsv"), sep='\t').values.tolist(), "test")

    def get_labels(self):
        """See base class."""
        if self.task_name == "sst-2" or self.task_name == "mr" or self.task_name == "cr" or self.task_name == "yelp-2":
            return list(range(2))
        elif self.task_name == "sst-5" or self.task_name == "yelp-5":
            return list(range(5))
        elif self.task_name == "agnews":
            return list(range(4))
        elif self.task_name == "yahoo":
            return list(range(10))
        elif self.task_name == "dbpedia":
            return list(range(14))
        elif self.task_name == "subj":
            return list(range(2))
        elif self.task_name == "trec":
            return list(range(6))
        else:
            raise Exception("task_name not supported.")
        
    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            if self.task_name in ['sst-2', 'mr', 'cr', 'yelp-2', 'sst-5', 'yelp-5', 'agnews', 'yahoo', 'dbpedia', 'subj', 'trec']:
                examples.append(InputExample(guid=guid, text_a=line[0], label=line[1]))
            else:
                raise Exception("Task_name not supported.")

        return examples
    

        
def text_classification_metrics(task_name, preds, labels):
    return {"acc": (preds == labels).mean()}

# Add your task to the following mappings

processors_mapping = {
    "sst-2": TextClassificationProcessor("sst-2"),
    "mr": TextClassificationProcessor("mr"),
    "cr": TextClassificationProcessor("cr"),
    "yelp-2": TextClassificationProcessor("yelp-2"),
    "sst-5": TextClassificationProcessor("sst-5"),
    "yelp-5": TextClassificationProcessor("yelp-5"),
    "agnews": TextClassificationProcessor("agnews"),
    "dbpedia": TextClassificationProcessor("dbpedia"),
    "yahoo": TextClassificationProcessor("yahoo"),
    "subj": TextClassificationProcessor("subj"),
    "trec": TextClassificationProcessor("trec"),
}

num_labels_mapping = {
    "sst-2": 2,
    "mr": 2,
    "cr": 2,
    "yelp-2": 2,
    "sst-5": 5,
    "yelp-5": 5, 
    "agnews": 4,
    "dbpedia": 14,    
    "yahoo": 10,
    "subj": 2,
    "trec": 6,    
}

output_modes_mapping = {
    "sst-2": "classification",
    "mr": "classification",
    "cr": "classification",
    "yelp-2": "classification",
    "sst-5": "classification",
    "yelp-5": "classification",
    "agnews": "classification",
    "dbpedia": "classification",
    "yahoo": "classification",
    "subj": "classification",
    "trec": "classification",
}

# Return a function that takes (task_name, preds, labels) as inputs
compute_metrics_mapping = {
    "sst-2": glue_compute_metrics,
    "mr": text_classification_metrics,
    "cr": text_classification_metrics,
    "yelp-2": text_classification_metrics,
    "sst-5": text_classification_metrics,
    "yelp-5": text_classification_metrics,
    "agnews": text_classification_metrics,
    "dbpedia": text_classification_metrics,
    "yahoo": text_classification_metrics,
    "subj": text_classification_metrics,
    "trec": text_classification_metrics,
}