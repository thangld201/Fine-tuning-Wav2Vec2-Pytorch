# Fine-tuning-Wav2Vec2-Pytorch
Fine-tuning a Wav2Vec2 model without reliance on Huggingface's Trainer.

```base``` folder contains a list of .txt and .wav files, in which a .wav file's transcript should be contained in the .txt file with the same name.
```main.sh``` bash script to fine-tune the model.<br>
```eval.sh``` bash script to evaluate the trained checkpoints on an evaluation dataset.

(This version of the code supports multi-gpu training)
