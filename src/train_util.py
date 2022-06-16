# train_util.py
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW
from transformers import Wav2Vec2ForCTC,Wav2Vec2Processor,get_linear_schedule_with_warmup
import os, torch
from tqdm.auto import tqdm

from data_util import DataCollatorCTCWithPadding
from eval_util import eval_model
from utils import dump_json

def init_processor():
    processor = Wav2Vec2Processor.from_pretrained('nguyenvulebinh/wav2vec2-base-vietnamese-250h')
    return processor
    
def init_model(device='cuda', processor = None):
    if not processor:
        processor = init_processor()
    model = Wav2Vec2ForCTC.from_pretrained(
        "nguyenvulebinh/wav2vec2-base-vietnamese-250h",
        ctc_loss_reduction="mean", 
        pad_token_id=processor.tokenizer.pad_token_id,
    )
    model.freeze_feature_encoder()
    model.gradient_checkpointing_enable()
    model.to(device)
    return model

def save_model(model, save_folder, nepoch=0, step=0):
    torch.save(model.state_dict(), os.path.join(save_folder,f"model_epoch{nepoch}_step{step}.pt"))

def load_model(model, save_checkpoint):
    model.load_state_dict(torch.load(save_checkpoint))

def trainloader(train_data, batch_size=32, data_collator=None, num_workers=4):
    return DataLoader(train_data, batch_size=batch_size, 
                      collate_fn=data_collator,sampler=RandomSampler(train_data),
                      num_workers=num_workers,pin_memory=True)
    
def testloader(test_data, batch_size=32, data_collator=None, num_workers=4):
    return DataLoader(test_data, batch_size=batch_size, 
                      collate_fn=data_collator,sampler=SequentialSampler(test_data),
                      num_workers=num_workers,pin_memory=True)

def train_n_epoch(model, train_data, batch_size=32,nepochs=10, num_workers=4,
                   validation_data=None,loggingstep=0.3,lr=1e-6, weight_decay=5e-4,
                   checkpoint_folder: str='.',data_collator: DataCollatorCTCWithPadding=None, resume_from_checkpoint: str=None):

    model.gradient_checkpointing_enable()

    trainLoader = trainloader(train_data,batch_size=batch_size, data_collator=data_collator,num_workers=num_workers)
    valLoader = testloader(validation_data,batch_size=batch_size, data_collator=data_collator,num_workers=num_workers)

    if resume_from_checkpoint is not None:
        load_model(model, resume_from_checkpoint)
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    total_train_step = int(nepochs*len(trainLoader))
    logstep = int(len(trainLoader)*loggingstep)
    warmup_steps = int(len(trainLoader)*0.3)

    # init optimizer & scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, weight_decay=weight_decay,eps=1e-12)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_train_step
    )

    _,val_wer_dict, val_loss=eval_model(model,validation_data,data_collator=data_collator,batch_size=batch_size,processor=train_data.processor,num_processor=num_workers,return_loss=True)
    print("-"*30)
    print("Before training")
    print(f"Val Loss = {val_loss} and Val Wer = {val_wer_dict['wer_greedy']}")
    print()
    # start training
    # train_loss = 0
    # val_loss = 0
    global_step = 0
    model.zero_grad()
    total_train_loss = 0
    training_stats = []
    print("Start training ...")
    print()
    pbar = tqdm(total=total_train_step)
    for i in range(nepochs):
        # start an epoch
        print('='*30)
        print(f"Epoch {i+1} ..")
        model.train()
        tmp_train_loss = 0
        tmp_val_loss = 0

        # epoch_iterator = tqdm(trainLoader, desc="Iteration", position=0, leave=True)
        for i, batch in enumerate(trainLoader):
            inputs = {'input_values':batch["input_values"].to("cuda"),
                    'labels':batch["labels"].to("cuda")}
            loss = model(**inputs).loss
            loss.backward()
            tmp_train_loss+=loss.item()
            total_train_loss+=loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            model.zero_grad()
            if (global_step!=0 and global_step%logstep==0 and validation_data is not None):
                _,val_wer_dict, val_loss=eval_model(model,validation_data,data_collator=data_collator,batch_size=batch_size,processor=train_data.processor,num_processor=num_workers,return_loss=True)
                print("-"*30)
                print(f"Step {global_step}")
                print(f"Val Loss = {val_loss} and Val Wer = {val_wer_dict['wer_greedy']}")
                save_model(model=model, save_folder=checkpoint_folder, nepoch=i+1, step=global_step)
                # save_model()
                training_stats.append({"step":global_step,
                                       "trainloss":total_train_loss/global_step,
                                       "valloss":val_loss,
                                       "valwer":val_wer_dict['wer_greedy']})
                dump_json(training_stats,os.path.join(checkpoint_folder,"training_stats.json"))
            global_step +=1
            pbar.update(1)
        pbar.close()

        print(f"Training loss: {tmp_train_loss/len(trainLoader)}")
        print(f"Validation loss: {tmp_train_loss/len(valLoader)}")

    print("\nFinish training !")
    # print(f"Average Training loss: {train_loss/nepochs}")