# Prompted Text Style Transfer Example

We experiment on Yelp for sentiment and Shakespeare for authorship transfer, respectively. 
For Shakespeare, we test for [few-shot text style transfer](https://arxiv.org/abs/2010.03802) by only training on 100 examples per style. 
Specifically, we provide 3 different training sets and their corresponding style classifiers.

## Setup

In addition to the dependencies for RLPrompt, you can install the additional dependencies for text style transfer with
```
pip install -r requirements.txt
```

## Download Pretrained Style Classifiers

You can download the style classifiers for training by running the script below
```
python download_tst_classifiers.py \
    --model_name [yelp-train,
                  shakespeare-train-100-0,
                  shakespeare-train-100-1,
                  shakespeare-train-100-2]
```

The `100` and `0,1,2` for `shakespeare` refer to the training data size and the random seed we used to sample the training data, respectively. 

## Running Experiments (Basic)

After that, you can run the experiment with the command below. Below are the explanations for some important parameters: 
- `dataset_seed`: The random seed of Shakespeare training sets. Right now we have 3 static sets to choose from. Skip this for Yelp
- `direction`: `0_to_1` refers to negative-to-positive for Yelp and old-to-modern for Shakespeare, whereas `1_to_0` means the opposite
- `task_lm`: The pretrained LM used for text generation
- `lower_outputs`: Whether to manually set outputs to lower case. Due to dataset properties, we set it to `true` for Yelp and `false` for Shakespeare
- `random_seed`: Random seed for initialization and sampling
```
python run_tst.py \
    dataset=[yelp, shakespeare] \
    dataset_seed=[0, 1, 2 (optional)] \
    direction=[0_to_1, 1_to_0] \
    task_lm=[distilgpt2, gpt2, gpt2-medium, gpt2-large, gpt2-xl] \
    lower_outputs=[true, false] \
    random_seed=[any integer (optional)]
```
The checkpoints and output prompts are saved at `outputs/[experiment-date]/[experiment-time]`


## Using Your Own Data

To run text style transfer using your own data, you would need to provide the following:
1. A text classifier which defines the style
2. A function for reading the data 

### Style Classifier
Our code can technically work with any binary text classifier using the `transformers` library. For out-of-the-box models, [Huggingface Model Hub](https://huggingface.co/models?pipeline_tag=text-classification&sort=downloads) is a great place to start with. 
If you'd like to train your own model, [this tutorial](https://huggingface.co/docs/transformers/training) provides an excellent example for fine-tuning a pretrained LM to serve as classifier. 

Once you've decided on the style classifier, you may add its name (in case of Model Hub models) or path (in case of local models) to `style_classifier_dict` under `tst_helpers.py`, which allows it to be called by its `dataset` name. 

### Data Loading Function
We load our data using the `load_text_style_transfer_dataset()` function in the `tst_data_utils.py` file. You can follow the format to specify the input texts and target style labels. 

## Evaluation

### Download Pretrained Evaluation Models
You can download the evaluation style classifiers with the following command
```
python download_tst_classifiers.py --model_name [yelp_test, shakespeare_test_all]
```
And the fine-tuned language models for perplexity as below
```
python download_ppl_lms.py --model_name [yelp, shakespeare]
```
After that, you can run the evaluation script with the command below
```
cd evaluation
python run_eval.py \
    prompt_0_to_1=[optional, learned prompt for the 0_to_1 direction] \
    prompt_1_to_0=[optional, learned prompt for the 1_to_0 direction] \
    dataset=[yelp, shakespeare] \
    dataset_seed=[optional, 0, 1, or 2 if dataset == shakespeare] \
    task_lm=[distilgpt2, gpt2, gpt2-medium, gpt2-large, gpt2-xl]
```
The outputs will be saved at `evaluation/outputs/[evaluation-date]/[evaluation-time]`