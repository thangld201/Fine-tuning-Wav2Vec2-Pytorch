# Fine-tuning-Wav2Vec2-Pytorch
Fine-tuning a Wav2Vec2 model without reliance on Huggingface's Trainer.

```base``` folder contains a list of .txt and .wav files, in which a .wav file's transcript should be contained in the .txt file with the same name.
```main.sh``` bash script to fine-tune the model.<br>
```eval.sh``` bash script to evaluate the trained checkpoints on an evaluation dataset.
By default, this repository use a default Vietnamese checkpoint as in ```src/train_util.py```. Feel free to change the checkpoint depending on the language/model to be fine-tuned.
