import sys
style_transformer_path = 'absolute/path/to/style/transformer'
sys.path.insert(0, style_transformer_path)

from data import load_dataset
from models import StyleTransformer
from utils import tensor2text
from base_generator import BaseGenerator

import argparse
import re
import time
from tqdm import tqdm
import torch
torch.set_grad_enabled(False)

def preprocess(text):
    text = re.sub('\s{2,}', ' ', text)
    text = re.sub('(.*?)( )([\.,!?\'])', r'\1\3', text)
    text = re.sub('([a-z])( )(n\'t)', r'\1\3', text)
    text = re.sub('\$ \_', r'$_', text)
    text = re.sub('(\( )(.*?)( \))', r'(\2)', text)
    text = re.sub('(``)( )*(.*?)', r"``\3", text)
    text = re.sub('(.*?)( )*(\'\')', r"\1''", text)
    return text

def get_lengths(tokens, eos_idx):
    lengths = torch.cumsum(tokens == eos_idx, 1)
    lengths = (lengths == 0).long().sum(-1)
    lengths = lengths + 1 # +1 for <eos> token
    return lengths

def generate(sentence, vocab, generator, sample, sample_size, top_k=10, style='pos'): 
    assert style in ['pos', 'neg']
    
    eos_idx = vocab.stoi['<eos>']
    inp_lengths = get_lengths(sentence, eos_idx)
    target_styles = torch.full_like(sentence[:, 0], 1 if style=='pos' else 0)
    start = time.time()
    
    with torch.no_grad():
        _, output_sentences = generator(
            sentence, 
            None,
            inp_lengths,
            target_styles,
            generate=True,
            sample=sample,
            generate_size=sample_size,
            top_k=top_k,
            differentiable_decode=False,
            temperature=1.0,
        )
    use_time = time.time() - start
    
    input_sentences = tensor2text(vocab, sentence.cpu()) * sample_size
    output_sentences = tensor2text(vocab, output_sentences.cpu())

    input_sentences = [preprocess(t) for t in input_sentences] 
    output_sentences  = [preprocess(t) for t in output_sentences] 
    
    return input_sentences, output_sentences

class StyleTransformerGenerator(): 
    def __init__(self, device_id): 
        parser = argparse.ArgumentParser()
    
        # evaluate
        parser.add_argument("--checkpoint",         default=style_transformer_path+'G-checkpoint.pth',    type=str, required=False)
        parser.add_argument("--batch_size",         default=1 ,           type=int, required=False)
        parser.add_argument("--sample_size",        default=1,            type=int, required=False)
        parser.add_argument("--random_seed",        default=2,           type=int, required=False)

        parser.add_argument("--use_dataset",        default='test',        type=str, required=False)    
        parser.add_argument("--use_dataset_size",   default=500,          type=int, required=False)    
        parser.add_argument("--use_content_reward", default='ctc',        type=str, required=False)  
        parser.add_argument("--use_style_reward",   default='open',       type=str, required=False)  
        parser.add_argument("--use_gpu",            default=1,            type=int, required=False)

        # data
        parser.add_argument("--data_path",  default=style_transformer_path+"data/yelp/")
        parser.add_argument("--min_freq",   default=3, type=int)
        parser.add_argument("--max_length", default=16, type=int)

        # model
        parser.add_argument("--load_pretrained_embed", help="whether to load pretrained embeddings.", action="store_true")
        parser.add_argument("--use_gumbel", help="handle discrete part in another way", action="store_true")
        parser.add_argument("-discriminator_method", help="the type of discriminator ('Multi' or 'Cond')", default="Multi")
        parser.add_argument("-embed_size", help="the dimension of the token embedding", default=256, type=int)
        parser.add_argument("-d_model", help="the dimension of Transformer d_model parameter", default=256, type=int)
        parser.add_argument("-head", help="the number of Transformer attention heads", dest="h", default=4, type=int)
        parser.add_argument("-num_styles", help="the number of styles for discriminator", default=2, type=int)
        parser.add_argument("-num_layers", help="the number of Transformer layers", default=4, type=int)
        parser.add_argument("-dropout", help="the dropout factor for the whole model", default=0.1, type=float)
        parser.add_argument("-learned_pos_embed", help="whether to learn positional embedding", default="1", choices=['1', '0', 'True', 'False'])
        parser.add_argument("-inp_drop_prob", help="the initial word dropout rate", default=0.1, type=float)
        parser.add_argument("-temp", help="the initial softmax temperature", default=1.0, type=float)
        self.args = parser.parse_args('')
        
        self.device = device_id
        self._load_data()
        self.generator = self._load_model(self.args, self.vocab, self.device)
        
    def _load_model(self, args, vocab, device): 
        generator = StyleTransformer(args, vocab).to(device)
        generator.load_state_dict(torch.load(args.checkpoint))
        generator.eval()
        return generator
        
    def _load_data(self): 
        # Test now on the real test set
        _, dev_iters, test_iters, vocab = load_dataset(self.args)
        sentence_dict = {}
        sentence_dict['src_pos2neg'] = test_iters.pos_iter
        sentence_dict['src_neg2pos'] = test_iters.neg_iter
        
        self.sentence_dict = sentence_dict
        self.vocab = vocab
        
    def sample_generate(self, 
                        task_name, 
                        sample_size=32, 
                        top_k=10, 
                        top_p=1.0, 
                        **kwargs): 
        # Assume sample_generate() function already checked that task_name is valid
        data_iter = self.sentence_dict[f'src_{task_name}']
        
#         target_label_dict = {'pos2neg': 'LABEL_0', 'neg2pos': 'LABEL_1'}
#         target_label = target_label_dict[task_name]
        target_style = task_name[-3:]
        
        all_outputs = []
        for i, example in tqdm(enumerate(data_iter), total=len(data_iter)):
            input_sentences, output_sentences = generate(example.text.to(self.device),
                                                         self.vocab,
                                                         self.generator,
                                                         sample=True,
                                                         sample_size=sample_size,
                                                         top_k=top_k,
                                                         style=target_style)
            
            all_outputs.append({'src': input_sentences[i],
                                'hypos': output_sentences,
                                'idx': i})
            
    def sample_and_save(self,
                        task_name,
                        save_path, 
                        **kwargs):
        all_outputs = self.sample_generate(task_name, 
                                            **kwargs)
        
        with open(save_path, 'w') as fw: 
            for row in all_outputs: 
                fw.write(json.dumps(row) + '\n')

if __name__ == '__main__':
    fire.Fire(StyleTransformerGenerator)