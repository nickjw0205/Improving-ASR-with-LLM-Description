from transformers_prompt import WhisperFeatureExtractor, WhisperTokenizer
from typing import Any, Dict, List, Union
import torch
from dataclasses import dataclass
import evaluate
from jiwer import process_words, wer_default
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import os

import re
import unicodedata
import regex

from jiwer import transforms as tr
from jiwer.transformations import wer_default, cer_default
from itertools import chain

import rapidfuzz
from rapidfuzz.distance import Opcodes

# tokenizer = WhisperTokenizer.from_pretrained('openai/whisper-base', language='en', task='transcribe')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# non-ASCII letters that are not separated by "NFKD" normalization
ADDITIONAL_DIACRITICS = {
    "œ": "oe",
    "Œ": "OE",
    "ø": "o",
    "Ø": "O",
    "æ": "ae",
    "Æ": "AE",
    "ß": "ss",
    "ẞ": "SS",
    "đ": "d",
    "Đ": "D",
    "ð": "d",
    "Ð": "D",
    "þ": "th",
    "Þ": "th",
    "ł": "l",
    "Ł": "L",
}

def remove_symbols_and_diacritics(s: str, keep=""):
    """
    Replace any other markers, symbols, and punctuations with a space, and drop any diacritics (category 'Mn' and some
    manual mappings)
    """

    def replace_character(char):
        if char in keep:
            return char
        elif char in ADDITIONAL_DIACRITICS:
            return ADDITIONAL_DIACRITICS[char]

        elif unicodedata.category(char) == "Mn":
            return ""

        elif unicodedata.category(char)[0] in "MSP":
            return " "

        return char

    return "".join(replace_character(c) for c in unicodedata.normalize("NFKD", s))

def remove_symbols(s: str):
    """
    Replace any other markers, symbols, punctuations with a space, keeping diacritics
    """
    return "".join(" " if unicodedata.category(c)[0] in "MSP" else c for c in unicodedata.normalize("NFKC", s))

class BasicTextNormalizer:
    def __init__(self, remove_diacritics: bool = False, split_letters: bool = False):
        self.clean = remove_symbols_and_diacritics if remove_diacritics else remove_symbols
        self.split_letters = split_letters

    def __call__(self, s: str):
        s = s.lower()
        s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)  # remove words between brackets
        s = re.sub(r"\(([^)]+?)\)", "", s)  # remove words between parenthesis
        s = self.clean(s).lower()

        if self.split_letters:
            s = " ".join(regex.findall(r"\X", s, regex.U))

        s = re.sub(r"\s+", " ", s)  # replace any successive whitespace characters with a space

        return s
# prepare feature extractor, tokenizer
feature_extractor = WhisperFeatureExtractor.from_pretrained('openai/whisper-base')
tokenizer = WhisperTokenizer.from_pretrained('openai/whisper-base', language='Hindi', task='transcribe')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prepare_dataset(batch):
    # audio를 16kHZ로 load
    audio = batch['audio']
    # padding & trucation 적용,log-mel spectrogram으로 변환
    batch['input_features'] = feature_extractor(audio['array'], sampling_rate=audio['sampling_rate']).input_features[0]
    
    batch['labels'] = tokenizer(batch['sentence']).input_ids
    return batch

# define a data collator
@dataclass
class DataCollatorSpeechS2SWhitPadding:
    processor: Any
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{'input_features': feature['input_features']} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors='pt').to(device)

        if features[0]['prompt'].numel() > 0:
            prompt_features = [{'input_ids': feature['prompt']} for feature in features]
            label_features = [{'input_ids': feature['labels']} for feature in features]

            combined_feature = []
            for prompt, label in zip(prompt_features, label_features):
                prompt_ids = prompt['input_ids'].tolist() if isinstance(prompt['input_ids'], torch.Tensor) else prompt['input_ids']
                label_ids = label['input_ids'].tolist() if isinstance(label['input_ids'], torch.Tensor) else label['input_ids']
                
                combined_ids = prompt_ids + [50257] + label_ids
                combined_feature.append({'input_ids': combined_ids})

            labels_batch = self.processor.tokenizer.pad(combined_feature, return_tensors='pt').to(device)
            
            labels = labels_batch['input_ids'].masked_fill(labels_batch.attention_mask.ne(1), -100)
            
            if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
                labels = labels[:, 1:]

            # Ensure 'prompts' is a list of tensors and pad them to the same length
            prompts = [prompt['input_ids'].clone().detach() for prompt in prompt_features]
            max_len = max([prompt.size(0) for prompt in prompts])
            padded_prompts = [torch.nn.functional.pad(prompt, (0, max_len - prompt.size(0)), value=self.processor.tokenizer.pad_token_id) for prompt in prompts]
            
            # Stack the padded prompts
            batch['prompts'] = torch.stack(padded_prompts)
            
            batch['labels'] = labels

        else:
            label_features = [{'input_ids': feature['labels']} for feature in features]
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors='pt').to(device)
            
            labels = labels_batch['input_ids'].masked_fill(labels_batch.attention_mask.ne(1), -100)
            
            if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
                labels = labels[:, 1:]
            
            batch['labels'] = labels
            batch['prompts'] = None

        return batch

# metric    
metric = evaluate.load('wer')

def compute_wer(pred, args, prompts):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    normalizer = BasicTextNormalizer()
    tokenizer = WhisperTokenizer.from_pretrained(f'openai/whisper-{args.model}', language='en', task='transcribe')
    
    # label의 -100dmf pad token으로 변환
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    total_wer = 0
    results = []
    batch_size = args.per_device_eval_batch_size
    print("\n\nDone inference!")
    print("Start decoding and calculating WER...")
    
    cutted_label_ids = []
    cutted_pred_ids = []
    
    if len(prompts) != 0:
        for i in tqdm(range(0, len(pred_ids))):
            cutted_pred_ids.append(pred_ids[i][len(prompts[i][0])+1:])
            cutted_label_ids.append(label_ids[i][len(prompts[i][0])+1:])
    
    for i in tqdm(range(0, len(cutted_pred_ids), batch_size)):
        batch_pred_ids = cutted_pred_ids[i:i + batch_size]
        batch_label_ids = cutted_label_ids[i:i + batch_size]        

        pre_strs = tokenizer.batch_decode(batch_pred_ids, skip_special_tokens=True)
        label_strs = tokenizer.batch_decode(batch_label_ids, skip_special_tokens=True)
        # pre_strs, label_strs = zip(*[(normalizer(pred), normalizer(label)) for pred, label in zip(pre_strs, label_strs) if label != 'ignore_time_segment_in_scoring'])
        
        filtered_pre_strs = []
        filtered_label_strs = []

        for pred, label in zip(pre_strs, label_strs):
            if label != 'ignore_time_segment_in_scoring':
                # 'ignore_time_segment_in_scoring'이 아닌 경우에만 리스트에 추가
                filtered_pre_strs.append(normalizer(pred))
                filtered_label_strs.append(normalizer(label))

        # 최종적으로 필터링된 리스트를 다시 튜플로 변환
        if filtered_pre_strs and filtered_label_strs:
                pre_strs, label_strs = zip(*zip(filtered_pre_strs, filtered_label_strs))
        else:
            pre_strs, label_strs = (), ()
        results.extend(zip(label_strs, pre_strs))
        
    # 파일에 모든 결과를 한 번에 쓰기
    with open(os.path.join(args.output_dir, 'refs_and_pred.txt'), 'w') as f:
        for ref, pred in results:
            f.write(f'Ref:{ref}\n')
            f.write(f'Pred:{pred}\n\n')

    # WER 계산
    pre_strs = [pred for _, pred in results]
    label_strs = [ref for ref, _ in results]
    total_wer = 100 * metric.compute(predictions=pre_strs, references=label_strs)

    return {'wer': total_wer}

