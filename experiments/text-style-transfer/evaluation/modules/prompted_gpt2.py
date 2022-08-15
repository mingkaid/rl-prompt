from transformers import AutoTokenizer, pipeline
from torch.utils.data import Dataset
from tqdm import tqdm
import json
import fire

def postprocess_output(text, end_punct='"', start_punct=None):         
    try: 
        end = text.index(end_punct)
    except ValueError: 
        end = len(text)
    text = text[:end].strip()
    # return text    
    if start_punct is not None: 
        start = text.find(start_punct)
        while start >= 0: 
            text = text[start+1:].strip()
            start = text.find(start_punct)

    try: 
        end = text.index('.')
    except ValueError: 
        end = len(text)

    try: 
        end = min(end, text.index('!'))
    except ValueError: 
        end = end

    try: 
        end = min(end, text.index('?'))
    except ValueError: 
        end = end

    return text[:end+1].strip().lower()

class SampleDataset(Dataset):
    def __init__(self, x):
        self.samples = x
        
    def __getitem__(self,index):
        return self.samples[index]
    
    def __len__(self):
        return len(self.samples)

class PromptedGPT2Generator(): 
    def __init__(self, 
                 model_name, 
                 device_id,
                 tst_template=None,
                 end_punct=None,
                 start_punct=None):       
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, pad_token='<|endoftext|>')
        self.generator = pipeline("text-generation",
                                  model=model_name,
                                  tokenizer=tokenizer,
                                  device=device_id)
        
        self.start_punct = start_punct
        self.end_punct = end_punct if end_punct is not None else '"'
        self.tst_template = tst_template if tst_template is not None else '{prompt} "{sentence_1}" "'
        
        
    def sample_generate(self, 
                        prompt_path, 
                        src_path, 
                        sample_size=32,
                        top_k=10, 
                        top_p=1.0,
                        **kwargs): 
        prompts = [line.strip() for line in open(prompt_path)]
        src_sentences = [line.strip() for line in open(src_path)]
        print(len(prompts), len(src_sentences))
        formatted_prompts = [self.tst_template.format(prompt=p, sentence_1=s) for p, s in zip(prompts, src_sentences)]
        print(formatted_prompts[0])
        
        src_encoding = self.generator.tokenizer(src_sentences)
        src_lens = [len(e) for e in src_encoding['input_ids']]
        print(src_lens[:5])
        
        all_outputs = []
        for i, (formatted_prompt, src_len) in tqdm(enumerate(zip(formatted_prompts, src_lens)), 
                                                    total=len(formatted_prompts)): 
            output_samples = self.generator(formatted_prompt,
                                            pad_token_id=50256,
                                            do_sample=True,
                                            max_new_tokens=max(1.5 * src_len, src_len + 10),
                                            top_k=top_k,
                                            top_p=top_p,
                                            num_return_sequences=sample_size,
                                            temperature=1.0,
                                            # Only return generated text, without the prompt
                                            return_full_text=False,
                                            **kwargs)
            generated_texts = []
            for output in output_samples: 
                text = output["generated_text"]
                generated_texts.append(postprocess_output(text, 
                                                            end_punct=self.end_punct,
                                                            start_punct=self.start_punct))
            # print(generated_texts)
            all_outputs.append({'prompt': prompts[i],
                                'src': src_sentences[i],
                                'hypos': generated_texts,
                                'idx': i})
                
            
        return all_outputs
    
    def sample_and_save(self,
                        prompt_path, 
                        src_path, 
                        save_path,
                        one_by_one=False,
                        **kwargs):
        all_outputs = self.sample_generate(prompt_path, 
                                           src_path, 
                                           one_by_one,
                                           **kwargs)
        
        with open(save_path, mode) as fw: 
            for row in all_outputs: 
                fw.write(json.dumps(row) + '\n')

if __name__ == '__main__':
    fire.Fire(PromptedGPT2Generator)