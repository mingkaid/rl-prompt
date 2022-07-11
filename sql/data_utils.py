import nltk
import spacy
from tqdm import tqdm
from datasets import Dataset
from collections import Counter
from collections import defaultdict
from typing import Any, List, Dict, Tuple, Callable, Optional

spacy_nlp = spacy.load(
    "en_core_web_sm",
    exclude=["ner",
             "parser",
             "tok2vec",
             "tagger",
             "lemmatizer",
             "attribute_ruler"])


def spacy_tokenize(text: str, lower: bool) -> List[str]:
    # Spacy will have different tokenization when the string
    # is lower-cased before tokenization, so we do it after.
    if lower is True:
        return [token.text.lower() for token in spacy_nlp(text)]
    else:
        return [token.text for token in spacy_nlp(text)]


def create_examples_from_dataset(
        dataset: Dataset,
        s1_name: str,
        s2_name: str,
        tokenization: Optional[str] = None,
        s1_preprocess_fn: Optional[Callable[[Any], str]] = None,
        s2_preprocess_fn: Optional[Callable[[Any], str]] = None,
) -> List[Tuple[str, str]]:
    if tokenization is None:
        tokenization = "nltk"

    if tokenization not in ["spacy", "nltk"]:
        raise ValueError

    if s1_preprocess_fn is None:
        s1_preprocess_fn = lambda s: s

    if s2_preprocess_fn is None:
        s2_preprocess_fn = lambda s: s

    examples = []
    for row in tqdm(dataset):
        s1 = s1_preprocess_fn(row[s1_name])
        s2 = s2_preprocess_fn(row[s2_name])

        if tokenization == "spacy":
            s1 = spacy_tokenize(s1, lower=True)
            s2 = spacy_tokenize(s2, lower=True)

        if tokenization == "nltk":
            s1 = nltk.word_tokenize(s1.lower())
            s2 = nltk.word_tokenize(s2.lower())

        examples.append((
            " ".join(s1),
            " ".join(s2)))

    return examples


def create_examples_from_dataset_2(
        dataset: Dataset,
        s1_name: str,
        s2_name: str,
        tokenization: Optional[str] = None,
        s1_preprocess_fn: Optional[Callable[[Any], str]] = None,
        s2_preprocess_fn: Optional[Callable[[Any], str]] = None,
) -> List[Tuple[str, str]]:
    if tokenization is None:
        tokenization = "nltk"

    if tokenization not in ["spacy", "nltk"]:
        raise ValueError

    if s1_preprocess_fn is None:
        s1_preprocess_fn = lambda s: s

    if s2_preprocess_fn is None:
        s2_preprocess_fn = lambda s: s

    examples = []
    for row in tqdm(dataset):
        if not isinstance(row[s2_name], list):
            raise TypeError

        s1 = s1_preprocess_fn(row[s1_name])
        s2 = [s2_preprocess_fn(s) for s in row[s2_name]]

        if tokenization == "spacy":
            s1 = spacy_tokenize(s1, lower=True)
            s2 = [spacy_tokenize(s, lower=True) for s in s2]

        if tokenization == "nltk":
            s1 = nltk.word_tokenize(s1.lower())
            s2 = [nltk.word_tokenize(s.lower()) for s in s2]

        examples.extend([(
            " ".join(s1),
            " ".join(s))
            for s in s2])

    return examples


def build_vocabulary_from_examples(examples: List[Tuple[str, str]]) -> Counter:
    counter: Counter = Counter()
    for pair in examples:
        if len(pair) != 2:
            raise ValueError
        counter.update(pair[0].split())
        counter.update(pair[1].split())
    return counter


def collect_unique_examples_from_examples(
        examples: List[Tuple[str, str]],
) -> Tuple[List[Tuple[str, str]],
           Dict[str, List[str]]]:

    unique_dict = defaultdict(list)
    unique_examples = []

    for source, target in examples:
        unique_dict[source].append(target)

    for source, targets in unique_dict.items():
        # Just use one of the targets as a placeholder
        unique_examples.append((source, targets[0]))

    return unique_examples, unique_dict
