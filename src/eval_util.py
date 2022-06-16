# eval_util.py
import multiprocessing, jiwer
from torch.utils.data import DataLoader, SequentialSampler
from typing import List, Dict,Union
import kenlm, pyctcdecode, torch
from tqdm.auto import tqdm

from data_util import DataCollatorCTCWithPadding

def get_decoder_ngram_model(tokenizer, ngram_lm_path):
    vocab_dict = tokenizer.get_vocab()
    sort_vocab = sorted((value, key) for (key, value) in vocab_dict.items())
    vocab = [x[1] for x in sort_vocab][:-2]
    vocab_list = vocab
    vocab_list[tokenizer.pad_token_id] = ""
    vocab_list[tokenizer.unk_token_id] = ""
    # convert space character representation
    vocab_list[tokenizer.word_delimiter_token_id] = " "
    # specify ctc blank char index, since conventially it is the last entry of the logit matrix
    alphabet = pyctcdecode.Alphabet.build_alphabet(vocab_list, ctc_token_idx=tokenizer.pad_token_id)
    lm_model = kenlm.Model(ngram_lm_path)
    decoder = pyctcdecode.BeamSearchDecoderCTC(alphabet,
                                   language_model=pyctcdecode.LanguageModel(lm_model))
    return decoder

def eval_model(model,test_data,processor=None,decoder=None,num_processor=4,beam_width=500, 
               beam_prune_logp=-10.0,batch_size=32,data_collator: DataCollatorCTCWithPadding=None,return_loss=False) -> List[Union[Dict, float]]:
    if processor is None:
        processor = test_data.processor
    print("Evaluating...")
    greedy_result = {'pred_str':[],'text':[]}
    lm_result = {'pred_str':[],'text':[]}
    dataloader = DataLoader(test_data,sampler=SequentialSampler(test_data),collate_fn=data_collator,batch_size=batch_size,shuffle=False, pin_memory=True)
    model.eval()
    total_loss = 0
    for batch in tqdm(dataloader):
        with torch.no_grad():
            inputs = {'input_values':batch["input_values"].to("cuda"),
                    'labels':batch["labels"].to("cuda")}
            outputs = model(**inputs)
            logits, loss = outputs.logits, outputs.loss
            total_loss += loss.item()

        batch['labels'][batch['labels'] == -100] = processor.tokenizer.pad_token_id
        batch["text"] = processor.batch_decode(batch["labels"], group_tokens=False)

        if decoder is not None:
            with multiprocessing.get_context("fork").Pool(processes=num_processor) as pool:
                text_list = decoder.decode_batch(pool, logits.cpu().detach().numpy(),beam_width=beam_width,beam_prune_logp=beam_prune_logp)
            batch["lm"] = text_list
            lm_result['pred_str'].extend(batch['lm'])
            lm_result['text'].extend(batch['text'])
        
        batch['greedy'] = processor.batch_decode(torch.argmax(logits, dim=-1))
        greedy_result['pred_str'].extend(batch['greedy'])
        greedy_result['text'].extend(batch['text'])
    
    greedy_result['pred_str'] = [s.replace('<unk>','') for s in greedy_result['pred_str']]
    wer_greedy = jiwer.wer(truth=greedy_result['text'],hypothesis=greedy_result['pred_str'])
    if decoder is not None:
        wer_lm = jiwer.wer(truth=lm_result['text'],hypothesis=lm_result['pred_str'])
        if return_loss:
            return {'decode_greedy':greedy_result,'decode_lm':lm_result},\
                    {'wer_greedy':wer_greedy,'wer_lm':wer_lm},\
                    total_loss/len(dataloader)
        else:
            return {'decode_greedy':greedy_result,'decode_lm':lm_result},\
                    {'wer_greedy':wer_greedy,'wer_lm':wer_lm}
    if return_loss:
        return {'decode_greedy':greedy_result},{'wer_greedy':wer_greedy}, total_loss/len(dataloader)
    return {'decode_greedy':greedy_result},{'wer_greedy':wer_greedy}