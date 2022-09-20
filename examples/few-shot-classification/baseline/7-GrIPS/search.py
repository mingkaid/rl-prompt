import nltk
import sys
from nltk.tokenize import word_tokenize, sent_tokenize
from supar import Parser
import string
import random
from nltk.tokenize.treebank import TreebankWordDetokenizer
import numpy as np
import argparse
# from nat_inst_gpt2 import *
from sklearn.metrics import balanced_accuracy_score
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from scipy.stats import entropy
import json
import torch
import os
torch.set_grad_enabled(False)


sys.path.append("..")
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer, 
    AutoModelForMaskedLM
)
from utils.dataset import FewShotDataset
from utils.processors import compute_metrics_mapping
import warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser(description='Take arguments from commandline')
parser.add_argument('--mode', default="Instruction Only", help='Type mode of instructions/prompts')
parser.add_argument('--num-shots', default=2, type=int, help='Type number of examples in the prompt if applicable')
parser.add_argument('--batch-size', default=2, type=int, help='Type in the batch-size')
parser.add_argument('--task-idx', default=1, type=int, help='Type in the index of the task based on the array in the code')
parser.add_argument('--seed', default=42, type=int, help='Type in seed that changes sampling of examples')
parser.add_argument('--train-seed', default=42, type=int, help='Type in seed that changes the sampling of edit operations (search seed)')
parser.add_argument('--num-compose', default=1, type=int, help='Number of edits composed to get one candidate')
parser.add_argument('--num-train', default=100, type=int, help='Number of examples in score set')
parser.add_argument('--level', default="phrase", help='level at which edit operations occur')
parser.add_argument('--simulated-anneal', action='store_true', default=False, help='runs simulated anneal if candidate scores <= base score')
parser.add_argument('--agnostic', action='store_true', default=False, help='uses template task-agnostic instruction')
parser.add_argument('--print-orig', action='store_true', default=False, help='print original instruction and evaluate its performance')
parser.add_argument('--write-preds', action='store_true', default=False, help='store predictions in a .json file')
parser.add_argument('--meta-dir', default='logs/', help='folder location to store metadata of search')
parser.add_argument('--meta-name', default='search.txt', help='file name to store metadata of search')
parser.add_argument('--patience', default=2, type=int, help='Type in the max patience P (counter)')
parser.add_argument('--num-candidates', default=5, type=int, help='Number of candidates in each iteration (m)')
parser.add_argument('--num-iter', default=10, type=int, help='Max number of search iterations')
parser.add_argument('--key-id', default=0, type=int, help='Use if you have access to multiple Open AI keys')
parser.add_argument('--edits', nargs="+", default=['del', 'swap', 'sub', 'add'], help='space of edit ops to be considered')


parser.add_argument('--task', type=str, required=True, help='data path')
parser.add_argument('--template', type=str, help='Template string')
parser.add_argument('--instruction', type=str, default=None, help='JSON object defining label map')
parser.add_argument('--label-map', type=str, default=None, help='JSON object defining label map')
parser.add_argument('--truncate_head', type=bool, default=True)
parser.add_argument('--skip_u0120', type=bool, default=False)
parser.add_argument('--gpu_id', type=int, default=0)
args = parser.parse_args()


instruction = args.instruction
template = args.template
device = torch.device(f'cuda:{args.gpu_id}')
args.task_name = args.task.lower()
args.mapping = args.label_map
args.data_dir = f"../../data/16-shot/{args.task_name}/16-{args.seed}"
args.use_demo = False
args.prompt = True
args.max_seq_length = 512
args.overwrite_cache = None
args.first_sent_limit = None
args.other_sent_limit = None
print(args)

seed = args.seed 
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

tokenizer = AutoTokenizer.from_pretrained('roberta-large')
generator = AutoModelForMaskedLM.from_pretrained('roberta-large').to(device)



os.makedirs(os.path.join('result', args.task), exist_ok=True)
meta_path = os.path.join('result', args.task, f'result-{args.seed}.txt')
meta_file = open(meta_path, 'w+')
batch_size = args.batch_size
num_shots = args.num_shots
mode = args.mode
seed = args.seed


parser = Parser.load('crf-con-en')
num_compose = args.num_compose
num_candidates = args.num_candidates
num_steps = args.num_iter
T_max = 10
edit_operations = args.edits
use_add = 'add' in edit_operations

if 'sub' in edit_operations:
    para_model_name = 'tuner007/pegasus_paraphrase'
    para_tokenizer = PegasusTokenizer.from_pretrained(para_model_name)
    para_model = PegasusForConditionalGeneration.from_pretrained(para_model_name).to(device).eval()



def score(candidate, split='train'):

    args.template = template.format(instruction=candidate)

    dataset = FewShotDataset(args, tokenizer=tokenizer, mode=split)
    metrics_fn = compute_metrics_mapping[args.task_name]
    
    dataloader = DataLoader(dataset, collate_fn=dataset.collate_fn, num_workers=4, pin_memory=True, batch_size=32, shuffle=False)
    pred_labels, true_labels = [], []
    for batch in dataloader:
        with torch.no_grad():
            logits = generator(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
            ).logits.cpu()

        logits = logits[range(logits.shape[0]), batch['mask_pos'].squeeze()]
        pred_labels += logits[:, dataset.get_labels_tok()].argmax(1).tolist()
        true_labels += batch['labels'].squeeze().tolist()

    metrics = metrics_fn(args.task_name, np.array(pred_labels), np.array(true_labels))
        
    if 'f1' in metrics:
        return metrics['f1']
    elif 'acc' in metrics:
        return metrics['acc']
    else:
        return metrics.values()[0]
    
    
    
def detokenize(tokens):
    return TreebankWordDetokenizer().detokenize(tokens)


def traverse_tree(parsed_tree):
    phrases = []
    for tree in parsed_tree:
        if tree.label() == '_': continue
        phrases.append(detokenize(tree.leaves()))
        for subtree in tree:
            if type(subtree) == nltk.tree.Tree:
                if subtree.label() == '_': continue
                phrases.append(detokenize(subtree.leaves()))
                phrases.extend(traverse_tree(subtree))
    return phrases

def check_child(tree):
    check = False
    count = 0
    total_count = 0
    for subtree in tree:
        total_count += 1
        if type(subtree) == nltk.tree.Tree:
            if subtree.label() == '_':
                count += 1
    if count >= total_count - count: check = True

    return check

def collect_leaves(parsed_tree):
    leaves = []
    for tree in parsed_tree:
        if type(parsed_tree) != nltk.tree.Tree: continue
        if tree.label() == '_': 
            leaves.append(detokenize(tree.leaves()))
            continue
        if check_child(tree): leaves.append(detokenize(tree.leaves()))
        else:
            leaves.extend(collect_leaves(tree))
    return leaves

def get_phrases(instruction): # one possible way of obtaining disjoint phrases
    phrases = []
    for sentence in sent_tokenize(instruction):
        parsed_tree = parser.predict(word_tokenize(sentence), verbose=False).sentences[0].trees[0]
        leaves = collect_leaves(parsed_tree)
        phrases.extend(leaves)
    phrases = [detokenize(word_tokenize(phrase)) for phrase in phrases if phrase not in string.punctuation or phrase == '']
    return phrases

def get_response(input_text,num_return_sequences,num_beams):
    batch = para_tokenizer([input_text],truncation=True,padding='longest',max_length=60, return_tensors="pt").to(device)
    translated = para_model.generate(**batch,max_length=60,num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=1.5)
    tgt_text = para_tokenizer.batch_decode(translated, skip_special_tokens=True)
    return tgt_text


def delete_phrase(candidate, phrase):
    if candidate.find(' ' + phrase) > 0:
        answer = candidate.replace(' ' + phrase, ' ')
    elif candidate.find(phrase + ' ') > 0:
        answer = candidate.replace(phrase + ' ', ' ')
    else: 
        answer = candidate.replace(phrase, '')
    return answer

def add_phrase(candidate, phrase, after):
    if after == '': answer = phrase + ' ' + candidate
    else: 
        if candidate.find(' ' + after) > 0:
            answer = candidate.replace(' ' + after, ' ' + after + ' ' + phrase)
        elif candidate.find(after + ' ') > 0:
            answer = candidate.replace(after + ' ', after + ' ' + phrase + ' ')
        else: 
            answer = candidate.replace(after, after + phrase )
    return answer

def swap_phrases(candidate, phrase_1, phrase_2):
    if candidate.find(' ' + phrase_1 + ' ') >= 0 : 
        answer = candidate.replace(' ' + phrase_1 + ' ', ' <1> ')
    else: answer = candidate.replace(phrase_1, '<1>')
    if candidate.find(' ' + phrase_2 + ' ') >= 0 : 
        answer = candidate.replace(' ' + phrase_2 + ' ', ' <2> ')
    else: answer = candidate.replace(phrase_2, '<2>')
    answer = answer.replace('<1>', phrase_2)
    answer = answer.replace('<2>', phrase_1)
    return answer

def substitute_phrase(candidate, phrase):
    num_beams = 10
    num_return_sequences = 10
    paraphrases = get_response(phrase, num_return_sequences, num_beams)
    paraphrase = np.random.choice(paraphrases, 1)[0] 
    paraphrase = paraphrase.strip('.')
    if candidate.find(' ' + phrase) > 0:
        answer = candidate.replace(' ' + phrase, ' ' + paraphrase)
    elif candidate.find(phrase + ' ') > 0:
        answer = candidate.replace(phrase + ' ', paraphrase + ' ')
    else: 
        answer = candidate.replace(phrase, paraphrase)
    return answer

def perform_edit(edit, base, phrase_lookup, delete_tracker):
    if edit == 'del':
        [i] = np.random.choice(list(phrase_lookup.keys()), 1) 
        result = delete_phrase(base, phrase_lookup[i]), [i]
    elif edit == 'swap':
        try: [i, j] = np.random.choice(list(phrase_lookup.keys()), 2, replace=False) 
        except: [i, j] = np.random.choice(list(phrase_lookup.keys()), 2, replace=True) 
        result = swap_phrases(base, phrase_lookup[i], phrase_lookup[j]), [i, j]
    elif edit == 'sub':
        [i] = np.random.choice(list(phrase_lookup.keys()), 1) 
        result = substitute_phrase(base, phrase_lookup[i]), [i]
    elif edit == 'add':
        keys = list(phrase_lookup.keys())
        keys.append(-1)
        [i] = np.random.choice(keys, 1) 
        if i >= 0: after = phrase_lookup[i]
        else: after = ''
        if len(delete_tracker) == 0: return base, []
        phrase = np.random.choice(delete_tracker, 1)[0]
        result = add_phrase(base, phrase, after), [phrase]
    return result


def get_phrase_lookup(base_candidate):
    if args.level == 'phrase': phrase_lookup = {p:phrase for p, phrase in enumerate(get_phrases(base_candidate))}
    elif args.level == 'word': 
        words = word_tokenize(base_candidate)
        words = [w for w in words if w not in string.punctuation or w != '']
        phrase_lookup = {p:phrase for p, phrase in enumerate(words)}
    elif args.level == 'sentence':
        sentences = sent_tokenize(base_candidate)
        phrase_lookup = {p:phrase for p, phrase in enumerate(sentences)}
    elif args.level == 'span':
        phrases = []
        for sentence in sent_tokenize(base_candidate):
            spans_per_sentence = np.random.choice(range(2,5)) # split sentence into 2, 3, 4, 5 chunks
            spans = np.array_split(word_tokenize(sentence), spans_per_sentence)
            spans = [detokenize(s) for s in spans]
            phrases.extend(spans)
        phrase_lookup = {p:phrase for p, phrase in enumerate(phrases)}
    else: raise ValueError()
    return phrase_lookup
               

operations_tracker = []
base_candidate = detokenize(word_tokenize(instruction))
assert word_tokenize(base_candidate) == word_tokenize(instruction)
original_candidate = base_candidate
meta_file.write("Base Candidate:\t "+ original_candidate + '\n')
base_score = score(base_candidate)
base_score_test = score(base_candidate, 'test')
meta_file.write("Base Train Score:\t "+ str(base_score) + '\n')
meta_file.write("Base Test Score:\t "+ str(base_score_test) + '\n')
delete_tracker = []
patience_counter = 1
for i in range(num_steps):
    meta_file.write("Running step:\t " + str(i) + '\n')
    deleted = {}
    added = {}
    phrase_lookup = get_phrase_lookup(base_candidate)
    #if base_candidate == original_candidate:
        #for p in phrase_lookup.values(): print(p)
    if use_add: 
        if len(delete_tracker): 
            if 'add' not in edit_operations: edit_operations.append('add')
        else: 
            if 'add' in edit_operations: edit_operations.remove('add')
    if num_compose == 1:
        edits = np.random.choice(edit_operations, num_candidates)
    else: 
        edits = []
        for n in range(num_candidates):
            edits.append(np.random.choice(edit_operations, num_compose))
            
    # generate candidates
    candidates = []
    # print(edits, base_candidate, phrase_lookup, delete_tracker)
    for edit in edits:
        if isinstance(edit, str): 
            meta_file.write("Performing edit:\t "+ edit + '\n')
            candidate, indices = perform_edit(edit, base_candidate, phrase_lookup, delete_tracker)
            meta_file.write("Generated candidate:\t "+ candidate + '\n')
            candidates.append(candidate)
            if edit  == 'del': deleted[candidate] = [phrase_lookup[indices[0]]]
            if edit == 'add': 
                if len(indices): added[candidate] = indices
        else:
            meta_file.write(("Performing edit:\t "+ ' '.join(edit))+ '\n')
            old_candidate = base_candidate
            composed_deletes = []
            composed_adds = []
            for op in edit:
                phrase_lookup = get_phrase_lookup(old_candidate)
                new_candidate, indices = perform_edit(op, old_candidate, phrase_lookup, delete_tracker)
                if op  == 'del':  composed_deletes.append(phrase_lookup[indices[0]])
                if op == 'add': 
                    if len(indices): composed_adds.append(indices[0])
                old_candidate = new_candidate
            meta_file.write("Generated candidate:\t "+ new_candidate+ '\n')
            candidates.append(new_candidate)
            if 'del' in edit: deleted[new_candidate] = composed_deletes
            if 'add' in edit and len(composed_adds) > 0: added[new_candidate] = composed_adds

    
    # print('Base score', base_score)
    scores = []
    for c, candidate in enumerate(candidates):
        scores.append(score(candidate))
        # print('New score:', scores[-1])
        meta_file.write("Score for Candidate "+ str(c)+ ":\t "+ str(scores[-1])+ '\n')
    
    meta_file.write("\n")
    best_idx = np.argmax(scores)
    best_score = scores[best_idx]
    if best_score > base_score: 
        patience_counter = 1
        base_candidate = candidates[best_idx]
        base_score = best_score
        operations_tracker.append(edits[best_idx])
        meta_file.write("New Candidate Found"+ '\n')
        meta_file.write("New Candidate Index:\t "+ str(best_idx)+ '\n')
        meta_file.write("New Candidate:\t "+ base_candidate+ '\n')
        meta_file.write("New Candidate Score:\t "+ str(base_score)+ '\n')
        try: meta_file.write("New Candidate Edit:\t "+ edits[best_idx]+ '\n')
        except: meta_file.write("New Candidate Edit:\t "+ ' '.join(edits[best_idx])+ '\n')
        meta_file.write("\n")
        # print('New Base Candidate: ', base_candidate)
        if base_candidate in added.keys():
            # print('Notice! Prev tracker: ', delete_tracker)
            for chunk in added[base_candidate]: 
                try: delete_tracker.remove(chunk)
                except: pass
            # print('Notice! New tracker: ', delete_tracker)
        if base_candidate in deleted.keys():
            delete_tracker.extend(deleted[base_candidate])
        base_candidate = detokenize(word_tokenize(base_candidate))
        
    else: 
        patience_counter += 1
        
        if args.simulated_anneal:
            K = 5
            T = T_max * np.exp(-i/K)
            idx = np.argmax(scores)
            chosen_score = scores[idx]
            prob = np.exp((chosen_score - base_score)/ T)
            if np.random.binomial(1, prob): 
                print('\n')
                print('Update from simulated anneal')
                meta_file.write('Update from simulated anneal \n')
                base_candidate = candidates[idx]
                base_score = chosen_score
                print('New Base Candidate: '+ base_candidate)
                if base_candidate in added.keys():
                    print('Notice! Prev tracker: ', delete_tracker)
                    for chunk in added[base_candidate]: 
                        try: delete_tracker.remove(chunk)
                        except: pass
                    print('Notice! New tracker: ', delete_tracker)
                if base_candidate in deleted.keys():
                    delete_tracker.extend(deleted[base_candidate])
                base_candidate = detokenize(word_tokenize(base_candidate))
            else:
                if patience_counter > args.patience:
                    print('Ran out of patience')
                    meta_file.write('Ran out of patience \n')
                    break
                else: continue        
            

        else:
            if patience_counter > args.patience:
                print('Ran out of patience')
                meta_file.write('Ran out of patience \n')
                break
            else: continue      
            
meta_file.write('\n')
print('\nTesting .... ')
meta_file.write('Testing .... \n')
if args.print_orig:
    print('Original Instruction:\t', original_candidate)
    orig_score = score(original_candidate, 'test')
    print('Original Accuracy:\t', str(orig_score))
print(f'Seed: {args.seed}')
if base_candidate == original_candidate: 
    print('No viable candidate found!')
    meta_file.write('No viable candidate found!\n')
    exit()

searched_score = score(base_candidate, 'test')
print('Accuracy after search:\t', str(searched_score))
print('Instruction after search:\t', base_candidate)
print('Edit Operations:\t', operations_tracker)
meta_file.write('Instruction after search:\t'+ base_candidate+ '\n')
meta_file.write('Accuracy after search:\t'+ str(searched_score)+ '\n')
meta_file.write('Edit Operations:\t'+ ' '.join([str(o) for o in operations_tracker]) + '\n')



