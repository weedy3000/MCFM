import os
import shutil

import pandas as pd

data_path = './data/iemocap'
sessions = ['Session1','Session2','Session3']
raw_data1 = {'audio_id': [], 'text': [], 'label': []}


for sess in sessions:
    root_path = os.path.join(data_path, sess)
    sentence_wav_path = os.path.join(root_path, 'sentences', 'wav')
    text_path = os.path.join(root_path, 'dialog', 'transcriptions')
    label_path = os.path.join(root_path, 'dialog', 'EmoEvaluation')

    raw_data = {}

    for label_file in os.listdir(label_path):
        if os.path.isfile(os.path.join(label_path, label_file)):
            with open(os.path.join(label_path, label_file), 'r') as f:
                """
                    这里表达的是，文件读取第一行 看第一行如果有文件则进行保存对应的标签和结果 其中包含标注信息
                    直到文件最后读取结束
                    """
                while True:
                    line = f.readline()
                    if not line:
                        break
                    if line[0] == '[':
                        t = line.split()
                        wav_name = t[3]
                        emotion_lable = t[4]
                        if wav_name not in raw_data.keys():
                            raw_data[wav_name] = {}
                        raw_data[wav_name]['label'] = emotion_lable
                        shutil.move(os.path.join(sentence_wav_path, os.path.splitext(label_file)[0], wav_name + '.wav'),
                                    os.path.join(data_path, 'train', 'wav'))

    for file in os.listdir(text_path):
        with open(os.path.join(text_path, file), 'r') as f_text:
            contents = f_text.readlines()
            for line in contents:
                if line[0] != 'S':
                    continue
                line = line.replace('\n', '')
                wav, con = line.split(':')
                w, _ = wav.split()
                try:
                    raw_data[w]['content'] = con[1:]
                except:
                    continue

    for k in list(raw_data.keys()):
        raw_data1['audio_id'].append(k)
        raw_data1['label'].append(raw_data[k]['label'])
        raw_data1['text'].append(raw_data[k]['content'])
print('------------train----------------')
print(len(raw_data1['audio_id']))
print(len(raw_data1['label']))
print(len(raw_data1['text']))
df = pd.DataFrame(raw_data1)
df.to_csv(os.path.join(data_path, 'train', 'label.csv'))


sessions_dev = ['Session4']
raw_data1 = {'audio_id': [], 'text': [], 'label': []}


for sess in sessions_dev:
    root_path = os.path.join(data_path, sess)
    sentence_wav_path = os.path.join(root_path, 'sentences', 'wav')
    text_path = os.path.join(root_path, 'dialog', 'transcriptions')
    label_path = os.path.join(root_path, 'dialog', 'EmoEvaluation')

    raw_data = {}

    for label_file in os.listdir(label_path):
        if os.path.isfile(os.path.join(label_path, label_file)):
            with open(os.path.join(label_path, label_file), 'r') as f:
                """
                    这里表达的是，文件读取第一行 看第一行如果有文件则进行保存对应的标签和结果 其中包含标注信息
                    直到文件最后读取结束
                    """
                while True:
                    line = f.readline()
                    if not line:
                        break
                    if line[0] == '[':
                        t = line.split()
                        wav_name = t[3]
                        emotion_lable = t[4]
                        if wav_name not in raw_data.keys():
                            raw_data[wav_name] = {}
                        raw_data[wav_name]['label'] = emotion_lable
                        shutil.move(os.path.join(sentence_wav_path, os.path.splitext(label_file)[0], wav_name + '.wav'),
                                    os.path.join(data_path, 'dev', 'wav'))

    for file in os.listdir(text_path):
        with open(os.path.join(text_path, file), 'r') as f_text:
            contents = f_text.readlines()
            for line in contents:
                if line[0] != 'S':
                    continue
                line = line.replace('\n', '')
                wav, con = line.split(':')
                w, _ = wav.split()
                try:
                    raw_data[w]['content'] = con[1:]
                except:
                    continue

    for k in list(raw_data.keys()):
        raw_data1['audio_id'].append(k)
        raw_data1['label'].append(raw_data[k]['label'])
        raw_data1['text'].append(raw_data[k]['content'])
print('------------dev----------------')
print(len(raw_data1['audio_id']))
print(len(raw_data1['label']))
print(len(raw_data1['text']))
df = pd.DataFrame(raw_data1)
df.to_csv(os.path.join(data_path, 'dev', 'label.csv'))


sessions_test = ['Session5']
raw_data1 = {'audio_id': [], 'text': [], 'label': []}


for sess in sessions_test:
    root_path = os.path.join(data_path, sess)
    sentence_wav_path = os.path.join(root_path, 'sentences', 'wav')
    text_path = os.path.join(root_path, 'dialog', 'transcriptions')
    label_path = os.path.join(root_path, 'dialog', 'EmoEvaluation')

    raw_data = {}

    for label_file in os.listdir(label_path):
        if os.path.isfile(os.path.join(label_path, label_file)):
            with open(os.path.join(label_path, label_file), 'r') as f:
                """
                    这里表达的是，文件读取第一行 看第一行如果有文件则进行保存对应的标签和结果 其中包含标注信息
                    直到文件最后读取结束
                    """
                while True:
                    line = f.readline()
                    if not line:
                        break
                    if line[0] == '[':
                        t = line.split()
                        wav_name = t[3]
                        emotion_lable = t[4]
                        if wav_name not in raw_data.keys():
                            raw_data[wav_name] = {}
                        raw_data[wav_name]['label'] = emotion_lable
                        shutil.move(os.path.join(sentence_wav_path, os.path.splitext(label_file)[0], wav_name + '.wav'),
                                    os.path.join(data_path, 'test', 'wav'))

    for file in os.listdir(text_path):
        with open(os.path.join(text_path, file), 'r') as f_text:
            contents = f_text.readlines()
            for line in contents:
                if line[0] != 'S':
                    continue
                line = line.replace('\n', '')
                wav, con = line.split(':')
                w, _ = wav.split()
                try:
                    raw_data[w]['content'] = con[1:]
                except:
                    continue

    for k in list(raw_data.keys()):
        raw_data1['audio_id'].append(k)
        raw_data1['label'].append(raw_data[k]['label'])
        raw_data1['text'].append(raw_data[k]['content'])
print('------------test----------------')
print(len(raw_data1['audio_id']))
print(len(raw_data1['label']))
print(len(raw_data1['text']))
df = pd.DataFrame(raw_data1)
df.to_csv(os.path.join(data_path, 'test', 'label.csv'))