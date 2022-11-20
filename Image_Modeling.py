import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import glob
import urllib.request
from PIL import Image
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

codes = pd.read_csv('C:\\Users\\INFOSTAT-18\\Desktop\\project\\KOPIS\\공연코드.csv')
hitornot = pd.read_csv('C:\\Users\\INFOSTAT-18\\Desktop\\project\\KOPIS\\final.csv')
already = pd.read_csv('C:\\Users\\INFOSTAT-18\\Desktop\\project\\KOPIS\\already.csv')

merged = pd.merge(codes, hitornot, how='inner', left_on='공연코드_1', right_on='공연코드')
merged = merged[['공연코드_1', '공연코드_2', '흥행수치_1', '좌석수']]
merged = merged.dropna(axis=0)
merged = merged.drop_duplicates(['공연코드_1', '공연코드_2']).reset_index(drop=True)

#al = [post.split('.')[0] for post in already['posters'].values.tolist()]
#already['poster'] = al
#already = already['poster']

#inter = pd.merge(merged, already, how='outer', left_on='공연코드_2', right_on='poster', indicator=True)
#inter = inter.query('_merge == "both"').drop(columns=['_merge'])
#inter = inter[['공연코드_1', '공연코드_2', '흥행수치_1', '좌석수']].reset_index(drop=True)

#포스터 이미지 약 4000여장 수집
'''
for num, code in zip(tqdm(range(inter.shape[0])), inter['공연코드_2']):
  url = 'https://www.kopis.or.kr/por/db/pblprfr/pblprfrView.do?menuId=MNU_00020&mt20Id={}'.format(code)
  response = requests.get(url)
  soup = BeautifulSoup(response.text, 'html.parser')
  img = soup.find('div', {'class':'tu'}).find('img')['src']
  savename = 'C:\\Users\\INFOSTAT-18\\Desktop\\project\\KOPIS\\posters\\'+str(code)+'.jpg'
  urllib.request.urlretrieve('https://www.kopis.or.kr'+img, savename)
'''

#흥행수치 히스토그램
'''
criteria_1 = [i for i in inter['흥행수치_1'] if i <= 1]

fig, ax = plt.subplots(2, 1, figsize=(15, 13))
fig.subplots_adjust(hspace=0.2)

ax[0].hist(criteria_1)
ax[0].set_title('criteria_1')

ax[1].hist(criteria_1, bins=10, linewidth=0.5, log=True)
ax[1].set_title('criteria_1_loged')

plt.show()
'''

#흥행수치에 따른 레이블링
'''
list_1 = []
for k in merged['흥행수치_1']:
  if k <= 0.1:
    list_1.append(0)
  elif k <= 0.25:
    list_1.append(1)
  elif k <= 0.4:
    list_1.append(2)
  elif k <= 0.7:
    list_1.append(3)
  elif k <= 1:
    list_1.append(4)
  else:
    list_1.append(5)

merged['흥행수치_1_label'] = list_1
idx = merged[merged['흥행수치_1_label']==5].index
merged = merged.drop(idx).reset_index(drop=True)
'''

#train, test, validation 그룹 나누기 및 그룹 로컬에 파일로 저장
'''
from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(merged[['공연코드_1', '공연코드_2', '좌석수']],
                                                     merged['흥행수치_1_label'], test_size=0.2, shuffle=True,
                                                      stratify=merged['흥행수치_1_label'], random_state=34)

x_valid, x_test, y_valid, y_test = train_test_split(x_valid, y_valid, test_size=0.5, shuffle=True,
                                                    stratify=y_valid, random_state=32)

x_train.to_csv('C:\\Users\\INFOSTAT-18\\Desktop\\project\\KOPIS\\x_train.csv')
x_valid.to_csv('C:\\Users\\INFOSTAT-18\\Desktop\\project\\KOPIS\\x_valid.csv')
x_test.to_csv('C:\\Users\\INFOSTAT-18\\Desktop\\project\\KOPIS\\x_test.csv')
y_train.to_csv('C:\\Users\\INFOSTAT-18\\Desktop\\project\\KOPIS\\y_train.csv')
y_valid.to_csv('C:\\Users\\INFOSTAT-18\\Desktop\\project\\KOPIS\\y_valid.csv')
y_test.to_csv('C:\\Users\\INFOSTAT-18\\Desktop\\project\\KOPIS\\y_test.csv')

import shutil

img_dir = 'C:\\Users\\INFOSTAT-18\\Desktop\\project\\KOPIS\\'

x_list = [x_train, x_valid, x_test]
y_list = [y_train, y_valid, y_test]
x_list_str = ['train', 'valid', 'test']

for x, y, z in zip(x_list, y_list, x_list_str):
  for code, label in zip(x['공연코드_2'], y):
    if label == 0:
      origin = img_dir+'posters\\'+str(code)+'.jpg'
      dest = img_dir+'\\'+str(z)+'\\'+str(label)+'\\'+str(code)+'.jpg'
      shutil.copyfile(origin, dest)

    elif label == 1:
      origin = img_dir+'\\posters\\'+str(code)+'.jpg'
      dest = img_dir+str(z)+'\\'+str(label)+'\\'+str(code)+'.jpg'
      shutil.copyfile(origin, dest)

    elif label == 2:
      origin = img_dir+'\\posters\\'+str(code)+'.jpg'
      dest = img_dir+str(z)+'\\'+str(label)+'\\'+str(code)+'.jpg'
      shutil.copyfile(origin, dest)

    elif label == 3:
      origin = img_dir+'\\posters\\'+str(code)+'.jpg'
      dest = img_dir+str(z)+'\\'+str(label)+'\\'+str(code)+'.jpg'
      shutil.copyfile(origin, dest)

    else:
      origin = img_dir+'\\posters\\'+str(code)+'.jpg'
      dest = img_dir+str(z)+'\\'+str(label)+'\\'+str(code)+'.jpg'
      shutil.copyfile(origin, dest)
'''

x_train = pd.read_csv('C:\\Users\\INFOSTAT-18\\Desktop\\project\\KOPIS\\x_train.csv')
x_test = pd.read_csv('C:\\Users\\INFOSTAT-18\\Desktop\\project\\KOPIS\\x_test.csv')
y_train = pd.read_csv('C:\\Users\\INFOSTAT-18\\Desktop\\project\\KOPIS\\y_train.csv')
y_test= pd.read_csv('C:\\Users\\INFOSTAT-18\\Desktop\\project\\KOPIS\\y_test.csv')
x_valid = pd.read_csv('C:\\Users\\INFOSTAT-18\\Desktop\\project\\KOPIS\\x_valid.csv')
y_valid = pd.read_csv('C:\\Users\\INFOSTAT-18\\Desktop\\project\\KOPIS\\y_valid.csv')


#이미지 데이터 전처리 후 배열로 변환
image_w = 224
image_h = 336
pixels = image_h * image_w * 3
X = {'train':[], 'test':[], 'valid':[]}
Y = {'train':[], 'test':[], 'valid':[]}
categories = ["0", "1", "2", "3", "4"]

x_list = [x_train, x_test, x_valid]
y_list = [y_train, y_test, y_valid]
x_list_str = ['train',  'test', 'valid']

destination = 'C:\\Users\\INFOSTAT-18\\Desktop\\project\\KOPIS\\'

for list_str in x_list_str:
  for idx, cls in enumerate(categories):
    label = [0 for i in range(len(categories))]
    label[idx] = 1

    image_dir = destination + list_str +'\\' + cls
    files = glob.glob(image_dir+'\\*.jpg')
    print(list_str, cls, " 파일 길이 : ", len(files))

    for n, k in enumerate(files):
      files[n] = k.split('\\')[-1]
    files = sorted(files)

    for i, f in enumerate(files):
      f = destination + list_str + '\\' + cls + '\\' + str(f)
      img = Image.open(f)
      img = img.convert("RGB")
      img = img.resize((image_w, image_h))
      data = np.asarray(img)

      X[list_str].append(data)
      Y[list_str].append(label)

  print('ok', len(Y[list_str]))

X['train'] = np.array(X['train'])
Y['train'] = np.array(Y['train'])

X['test'] = np.array(X['test'])
Y['test'] = np.array(Y['test'])

X['valid'] = np.array(X['valid'])
Y['valid'] = np.array(Y['valid'])

#이미지 데이터 크기 정규화 후 vgg16 모델 로딩
X['train'] = X['train'].astype(float) / 255
X['test'] = X['test'].astype(float) / 255
X['valid'] = X['valid'].astype(float) / 255

from keras import layers, models
from keras.applications import VGG16
from keras import Input
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers, initializers, regularizers, metrics
from keras.callbacks import ModelCheckpoint
from PIL import Image
from keras.layers import Dense, GlobalAveragePooling2D

cnn = VGG16(include_top= False, weights='imagenet')

input_tensor = Input(shape=(336, 224, 3))
x = cnn(input_tensor)
x = GlobalAveragePooling2D()(x)
output_tensor = Dense(5, activation ='softmax')(x)

model = Model(input_tensor, output_tensor)

checkpoint = ModelCheckpoint(filepath='My_VGG_weight.hdf5',
                             monitor='loss',
                             mode='min',
                             save_best_only=True)

model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=1e-5), metrics=['acc'])
history = model.fit(X['train'], Y['train'], steps_per_epoch=25,
            epochs=40, validation_data=(X['valid'], Y['valid']),
            validation_steps=16, callbacks=[checkpoint])

model.save('C:\\Users\\INFOSTAT-18\\Desktop\\project\\KOPIS\\model.h5')