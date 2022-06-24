#eval_model.py
import multiprocessing, jiwer
from torch.utils.data import DataLoader, SequentialSampler
from typing import List, Dict,Union
import kenlm, pyctcdecode, torch
from tqdm.auto import tqdm
import argparse
from os import listdir
from os.path import join

from load_data import get_audio_txt
from data_util import DataCollatorCTCWithPadding, Speech2TextDataset
from eval_util import eval_model, get_decoder_ngram_model
from train_util import init_processor, init_model, load_model, testloader
from utils import dump_json, get_json


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_data_path", default=None, required=True, type=str, help="Path containing .wav and .txt files (for evaluation)")
    parser.add_argument("--max_workers",default=4,type=int,help="Max workers in dataloader")
    parser.add_argument("--chunksize",default=128, type=int,help="Chunksize in loading data")
    parser.add_argument("--lm_path",default=None,type=str,help="Path to language model file")
    parser.add_argument("--checkpoint_folder",default='./checkpoint',type=str,help="Path to folder containing checkpoint (.pt) files (evaluation)")
    parser.add_argument("--eval_result_save_folder",default='./checkpoint',type=str,help="Folder to write decode results and wer result file (.json)")
    parser.add_argument("--batch_size",default=32,type=int)
    args = parser.parse_args()
    
    min_duration = -1.0
    max_duration = 1e9

    model = init_model("cuda")
    model=torch.nn.DataParallel(model)
    model.to("cuda")
    processor = init_processor()
    print("Finish loading base model...")
    
    audio_list, text_list = get_audio_txt(args.eval_data_path,
                                          min_duration=min_duration,
                                          max_duration = max_duration, 
                                          max_workers=args.max_workers, 
                                          chunksize=args.chunksize)

    test_dataset = Speech2TextDataset(audio_list_path=audio_list, 
                                      text_list=text_list,
                                      processor=processor,
                                      max_duration=max_duration,
                                      min_duration=min_duration)

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    print("Finish loading raw data...")
    decoder = get_decoder_ngram_model(processor.tokenizer, args.lm_path)
    print("Finish loading language model decode...")
    result = {}
    print("Evaluating base model...")
    val_decode_dict,val_wer_dict, val_loss=eval_model(model,test_dataset,
                                        decoder=decoder,
                                        data_collator=data_collator,
                                        batch_size=args.batch_size,
                                        processor=test_dataset.processor,
                                        num_processor=args.max_workers,
                                        return_loss=True)
    result['base'] = {'decode':val_decode_dict,
            'wer':val_wer_dict,
            'loss':val_loss}
    print(f"Base model's val loss is {val_loss}")
    print(f"Base model's wer is {val_wer_dict['wer_greedy']} (AM) and {val_wer_dict['wer_lm']} (LM)")
    for f in listdir(args.checkpoint_folder):
        print("-"*30)
        print(f"Loading checkpoint from {f}")
        load_model(model, join(args.checkpoint_folder,f))
        val_decode_dict,val_wer_dict, val_loss=eval_model(model,test_dataset,
                                            decoder=decoder,
                                            data_collator=data_collator,
                                            batch_size=args.batch_size,
                                            processor=test_dataset.processor,
                                            num_processor=args.max_workers,
                                            return_loss=True)
        print(f"Val loss is {val_loss}")
        print(f"Wer AM: {val_wer_dict['wer_greedy']}")
        print(f"Wer LM: {val_wer_dict['wer_lm']}")
        result[f] = {'decode':val_decode_dict,
                    'wer':val_wer_dict,
                    'loss':val_loss}
        dump_json(result,join(args.eval_result_save_folder,"evaluation_result.json"))
    print()
    print()
    print()
    print("Finish Evaluation !")
