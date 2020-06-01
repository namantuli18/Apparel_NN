import cv2
import numpy as np
from tqdm import tqdm
from random import shuffle
import pandas as pd
import os
import matplotlib.pyplot as plt
import pickle
LR=1e-3
MODEL_NAME='r'
dataset=["T-Shirt",
	"Trouser",
	"Pullover",
	"Dress",
	"Coat",
	"Sandal",
	"Shirt",
	"Sneaker",
	"Bag",
	"Ankle boot"]
df=pd.read_csv(r'train.csv')
#MODEL_NAME='ageprediction.model'.format(LR,'2conv-basic-video')

#print(df.head())
TRAIN_DIR=r"train"
IMG_SIZE=28
TEST_DIR=r"test"
def return_class_from_img(df,img):
	array_ret=np.zeros(10,dtype=int)	
	for x,i in enumerate(df['id']):
		#print(array_ret)
		if str(i)==img:
			#print(img)
			img_label= df['label'][x]
			array_ret[img_label]=1
			return array_ret
			#print(img_label)
		
#print(return_class_from_img(df,'10'))			
#print(df.head())			
'''for img in tqdm(os.listdir(TRAIN_DIR)):
	img_num=img.split(".")[0]
	#print(img)
	label=return_class_from_img(df,img_num)
	print(label)
'''

def train_images():
	training_data=[]
	for img in tqdm(os.listdir(TRAIN_DIR)):
		img_num=img.split(".")[0]
		label=return_class_from_img(df,img_num)
		path=os.path.join(TRAIN_DIR,img)
		img=cv2.resize(cv2.imread(path,cv2.IMREAD_UNCHANGED),(IMG_SIZE,IMG_SIZE))
		training_data.append([np.array(img),np.array(label)])
		
		#print('Processing {}.png'.format(img_num))
	shuffle(training_data)

	np.save('train_data1.npy',training_data)
	return training_data
#train_data=train_images()
train_data=np.load("train_data.npy",allow_pickle=True)
def test_images():
	testing_data=[]
	for img in tqdm(os.listdir(TEST_DIR)):
		img_num=img.split('.')[0]
		path =os.path.join(TEST_DIR,img)
		img=cv2.resize(cv2.imread(path,cv2.IMREAD_UNCHANGED),(IMG_SIZE,IMG_SIZE))
		testing_data.append([np.array(img),np.array(img_num)])
		#print("Processing {}".format(img_num))
	np.save('testing_data1.npy',testing_data)
	return testing_data
#test_data=test_images()
test_data=np.load("testing_data.npy",allow_pickle=True)
import tflearn
from tflearn.layers.conv import max_pool_2d,conv_2d
from tflearn.layers.core import input_data,fully_connected,dropout
from tflearn.layers.estimator import regression


convnet=input_data(shape=[None,IMG_SIZE,IMG_SIZE,1],name='input')

convnet=conv_2d(convnet,32,10,activation='relu')
convnet=max_pool_2d(convnet,10)
convnet=conv_2d(convnet,64,10,activation='relu')
convnet=max_pool_2d(convnet,10)
convnet=conv_2d(convnet,32,10,activation='relu')
convnet=max_pool_2d(convnet,10)
convnet=conv_2d(convnet,64,10,activation='relu')
convnet=max_pool_2d(convnet,10)
convnet=conv_2d(convnet,32,10,activation='relu')
convnet=max_pool_2d(convnet,10)
convnet=conv_2d(convnet,64,10,activation='relu')
convnet=max_pool_2d(convnet,10)




convnet=fully_connected(convnet,1024,activation='relu')
convnet=dropout(convnet,0.8)


convnet = fully_connected(convnet, 10, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet,tensorboard_dir='log')
if os.path.exists('{}.meta'.format(MODEL_NAME)):
	model.load(MODEL_NAME)
	print('Model Loaded')


train=train_data[:-500]
test=train_data[-500:]
X=np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
Y=[i[1] for i in train]

test_x=np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)

test_y=[i[1] for i in test]
#print(test_x)

model.fit({'input': X}, {'targets': Y}, n_epoch=5, validation_set=({'input': test_x}, {'targets': test_y}), 
    snapshot_step=500, show_metric=True, run_id='MODEL_NAME' )
with open('model.pickle','wb') as f:
	pickle.dump(model,f)

fig=plt.figure()
for num,data in enumerate(test_data[11:41]):
	img_num=data[1]
	img_data=data[0]
	y=fig.add_subplot(5,6,num+1)
	orig=img_data
	data=img_data.reshape(IMG_SIZE,IMG_SIZE,1)
	model_out=model.predict([data])[0]
	for count,items in enumerate(dataset):
		if count==np.argmax(model_out):
			str_label=dataset[count]

	
	y.imshow(orig,cmap='gray')
	plt.title(str_label)
	y.axes.get_xaxis().set_visible(False)
	y.axes.get_yaxis().set_visible(False)
plt.show()

'''
with open('submission1v02.csv','w') as f:
	f.write('id,label\n')
with open('submission1v02.csv','a') as f:
	for data in tqdm(test_data):
		img_num=data[1]
		img_data=data[0]
		#y=fig.add_subplot(3,4,num+1)
		orig=img_data
		data=img_data.reshape(IMG_SIZE,IMG_SIZE,1)
		model_out=model.predict([data])[0]
		#print(model_out)
		
		f.write(f'{img_num},{np.argmax(model_out)}\n')
'''