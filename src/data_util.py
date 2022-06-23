# data_util.py
import librosa, re, os, torch
from torch.utils.data import Dataset
from transformers import Wav2Vec2Processor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from tqdm.auto import tqdm

def get_duration(filename):
    y, _=librosa.load(filename,sr=16000)
    return librosa.get_duration(y=y,sr=16000)

class Speech2TextDataset(Dataset):
    def __init__(self, audio_list_path: List[str], text_list: List[str] = None, max_duration: int = None,
                 min_duration: int = None, processor: Wav2Vec2Processor = None):
        self.audio_list_path = audio_list_path
        self.text_list = text_list
        self.processor = processor
        self.chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'
        self.sampling_rate = 16000
        self.min_duration = min_duration
        self.max_duration = max_duration
        assert len(self.audio_list_path)==len(self.text_list), "Number of audios and texts are inconsistent!"
        # self.recheck_data()
        if self.text_list is not None:
            self.process_text()

    def __len__(self):
        return len(self.audio_list_path)

    def __getitem__(self, idx: int):
        input_values = self.read_audio(self.audio_list_path[idx])
        if self.text_list is not None:
            labels = self.text_ids[idx]
            return {'input_values':input_values,'labels':labels}
        return {'input_values':input_values}

    def preprocess_text(self, txt: str):
        txt = re.sub(self.chars_to_ignore_regex, '', txt).lower().strip()+" "
        txt = re.sub(' +', ' ',txt)
        return txt

    def process_text(self):
        """
        Labels to ids
        """
        self.text_list = [self.preprocess_text(txt) for txt in self.text_list]
        text_ids = []
        for txt in tqdm(self.text_list,desc='Encode label ids...'):
            with self.processor.as_target_processor():
                id_txt = self.processor(txt).input_ids
                text_ids.append(id_txt)
        self.text_ids = text_ids

    def read_audio(self, path: str):
        y, _=librosa.load(path,sr=self.sampling_rate)
        return self.processor(y, sampling_rate=self.sampling_rate).input_values[0]

    def recheck_data(self):
        durs = [self.get_duration(f) for f in self.audio_list_path]
        usable = [i for i in range(len(self.audio_list_path))]
        if self.min_duration is not None:
            usable = [i for i in usable if durs[i]>=self.min_duration]
        if self.max_duration is not None:
            usable = [i for i in usable if durs[i]<=self.max_duration]
        self.audio_list_path = [self.audio_list_path[i] for i in usable]
        if self.text_list is not None:
            self.text_list = [self.text_list[i] for i in usable]
        durs = [durs[i] for i in usable]
        print(f"Total durations of audios in dataset: {sum(durs)/3600} hours")
    
    def get_duration(self, path):
        y, _=librosa.load(path,sr=16000)
        return librosa.get_duration(y=y,sr=16000)

@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )
        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch
