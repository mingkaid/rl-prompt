import json
import fire

class BaselineGenerator(): 
    def copy_and_save(self,
                      src_path,
                      save_path): 
        src_sentences = [line.strip() for line in open(src_path)]
        
        all_outputs = [{'src': x, 
                        'hypos': [x],
                        'idx': i} for i, x in enumerate(src_sentences)]
        
        with open(save_path, 'w') as fw: 
            for row in all_outputs: 
                fw.write(json.dumps(row) + '\n')
                
    def ref_and_save(self,
                     src_path,
                     ref_path,
                     save_path): 
        srcs = [line.strip() for line in open(src_path)]
        refs = [line.strip() for line in open(ref_path)]
        all_outputs = [{'src': x, 
                        'hypos': [r],
                        'idx': i} for i, (x, r) in enumerate(zip(srcs, refs))]
        
        with open(save_path, 'w') as fw: 
            for row in all_outputs: 
                fw.write(json.dumps(row) + '\n')
                
if __name__ == '__main__':
    fire.Fire(BaselineGenerator)