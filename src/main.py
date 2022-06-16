from load_data import get_audio_txt
from data_util import Speech2TextDataset, DataCollatorCTCWithPadding
from train_util import init_model, init_processor, train_n_epoch
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=None, required=True, type=str, help="Path containing .wav and .txt files")
    parser.add_argument("--train_split", default=0.99, type=float, help="Percentage of data to train, the rest is used for validation")
    parser.add_argument("--max_duration",default=10.0, type=float, help="Only use files with duration below this threshold in training")
    parser.add_argument("--min_duration",default=2.0, type=float, help="Only use files with duration above this threshold in training")
    parser.add_argument("--max_workers",default=4,type=int,help="Max workers in dataloader")
    parser.add_argument("--chunksize",default=128, type=int,help="Chunksize in loading data")
    parser.add_argument("--resume_from_checkpoint",default=None,type=str,help="Start training from a certain model checkpoint")
    parser.add_argument("--checkpoint_folder",default='./checkpoint',type=str,help="Folder to save checkpoints and training stats")
    parser.add_argument("--logging_percent_per_epoch",default=0.3,type=float,help="For each % percent of an epoch, evaluate on validation data and save checkpoint")
    parser.add_argument("--lr",default=3e-5,type=float,help='Learning rate')
    parser.add_argument("--weight_decay",default=1e-4,type=float)
    parser.add_argument("--epoch",default=5,type=int,help="Number of epochs for training")
    parser.add_argument("--batch_size",default=32,type=int)
    # parser.add_argument()
    args = parser.parse_args()
    
    model = init_model("cuda")
    processor = init_processor()
    print("Finish loading model...")
    audio_list, text_list = get_audio_txt(args.base,min_duration=args.min_duration,
                                          max_duration = args.max_duration, 
                                          max_workers=args.max_workers, 
                                          chunksize=args.chunksize)

    split_id = int(args.train_split*len(audio_list))

    train_dataset = Speech2TextDataset(audio_list_path=audio_list[:split_id], 
                                       text_list=text_list[:split_id], 
                                       processor=processor,
                                       max_duration=args.max_duration,
                                       min_duration=args.min_duration)

    test_dataset = Speech2TextDataset(audio_list_path=audio_list[split_id:], 
                                      text_list=text_list[split_id:],
                                      processor=processor,
                                      max_duration=args.max_duration,
                                      min_duration=args.min_duration)

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    print("Finish loading raw data...")
    training_args = {
        'model':model,
        'train_data':train_dataset,
        'validation_data':test_dataset,
        'loggingstep':args.logging_percent_per_epoch, # %/epoch to evaluate performance
        'batch_size':args.batch_size,
        'nepochs':args.epoch,
        'lr':args.lr,
        'weight_decay':args.weight_decay,
        'checkpoint_folder':args.checkpoint_folder,
        'data_collator':data_collator,
        'resume_from_checkpoint':args.resume_from_checkpoint
    }

    train_n_epoch(**training_args)