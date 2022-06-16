# load_data.py
import librosa, os
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map

def audioid(f):
    return str(''.join(f.split('.')[:-1]))

def read(f):
    with open(f,'r',encoding='utf-8') as f:
        d = f.read().strip()
    return d

def readyt(base):
    audios = {}
    texts = {}
    for f in os.listdir(base):
        if f.endswith('.txt'):
            texts[audioid(f)]=read(os.path.join(base,f))
        else:
            audios[audioid(f)]=os.path.join(base,f)
    return audios, texts

def get_duration(filename):
    y, _=librosa.load(filename,sr=16000)
    return librosa.get_duration(y=y,sr=16000)

def process_sample(p):
    try:
        d = get_duration(p[0])
    except:
        return None
    if d<p[2] or d>p[3]:
        return None
    return {'audio':p[0],'text':p[1],'duration':d}

def get_audio_txt(base, txt_dict: dict=None, min_duration=2.0, max_duration = 10.0,
                  max_workers=4,chunksize=128):
    audios, texts = readyt(base)
    all_list = []
    for f in audios:
        txt = ''
        if f not in texts:
            if txt_dict is not None:
                if f not in txt_dict:
                    continue
                txt = txt_dict[f]
            else:
                continue
        else:
            txt = texts[f]
        all_list.append([audios[f],txt,min_duration,max_duration])
    data = process_map(process_sample,all_list,max_workers=max_workers,chunksize=chunksize)
    data = [p for p in data if p is not None]
    audio_list = [p['audio'] for p in data]
    text_list = [p['text'] for p in data]
    durs = [p['duration'] for p in data]
    print(f"Total duration {sum(durs)/3600}")
    return audio_list, text_list

# audios, texts = readyt('/content/audio_5h')
# audios2, texts2 = readyt('/content/audio_20h')