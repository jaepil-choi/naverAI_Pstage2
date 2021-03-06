import pickle as pickle
import os
import pandas as pd
import numpy as np
import torch

# Dataset 구성.
class RE_Dataset(torch.utils.data.Dataset):
  def __init__(self, tokenized_dataset: pd.DataFrame, labels: np.ndarray):
    self.tokenized_dataset = tokenized_dataset
    self.labels = labels

  def __getitem__(self, idx: int) -> dict:
    item = {key: torch.tensor(val[idx]) for key, val in self.tokenized_dataset.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)

# 처음 불러온 tsv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다.
# 변경한 DataFrame 형태는 baseline code description 이미지를 참고해주세요.
def preprocessing_dataset(dataset: pd.DataFrame, label_type: dict) -> pd.DataFrame:
  label = []
  for i in dataset[8]:
    if i == 'blind':
      label.append(100)
    else:
      label.append(label_type[i])
  out_dataset = pd.DataFrame({'sentence':dataset[1],'entity_01':dataset[2],'entity_02':dataset[5],'label':label,})
  return out_dataset

# tsv 파일을 불러옵니다.
def load_data(dataset_dir: str) -> pd.DataFrame:
  # load label_type, classes
  with open('/opt/ml/input/data/label_type.pkl', 'rb') as f:
    label_type = pickle.load(f)
  # load dataset
  dataset = pd.read_csv(dataset_dir, delimiter='\t', header=None)
  # preprecessing dataset
  dataset = preprocessing_dataset(dataset, label_type)
  
  return dataset

# bert input을 위한 tokenizing.
# tip! 다양한 종류의 tokenizer와 special token들을 활용하는 것으로도 새로운 시도를 해볼 수 있습니다.
# baseline code에서는 2가지 부분을 활용했습니다.
def tokenized_dataset(dataset: pd.DataFrame, tokenizer: 'AutoTokenizer') -> 'BatchEncoding':
  concat_entity = []
  for e01, e02 in zip(dataset['entity_01'], dataset['entity_02']):
    temp = ''
    temp = e01 + '[SEP]' + e02
    concat_entity.append(temp) # [('기아자동차', 'K5'), ...]
  tokenized_sentences = tokenizer(
      text=concat_entity,
      text_pair=list(dataset['sentence']),
      return_tensors="pt",
      padding=True,
      truncation='only_second', # tokenizer의 concat entity, dataset 리스트가 들어가는데, max len을 넘어가면 "앞에서부터" 자르기 때문에 entity가 짤림. 
      max_length=100,
      add_special_tokens=True,

      # return_token_type_ids=True, # Roberta large 돌릴 때는 이거 없으면 token_type_ids 없다고 한다. 근데 이거 추가하면 시간 엄청 길어짐. 
      )
  return tokenized_sentences
