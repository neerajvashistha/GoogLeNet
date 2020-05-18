# python train.py --dataset mnist --model vgg16 --reshape '(32,32)' --batch_size 128 --epoch 10 --learning_rate 0.01 --dropout_rate 0.2 --activation_ch softmax --optimizer_ch sgd
# python train.py --dataset cifar --model vgg16 --reshape '(32,32)' --batch_size 128 --epoch 10 --learning_rate 0.01 --dropout_rate 0.2 --activation_ch softmax --optimizer_ch sgd
# python train.py --dataset mnist --model googlenet --reshape '(224,224)' --batch_size 128 --epoch 10 --learning_rate 0.001 --dropout_rate 0.2 --activation_ch softmax --optimizer_ch sgd
# python train.py --dataset cifar --model googlenet --reshape '(224,224)' --batch_size 128 --epoch 10 --learning_rate 0.001 --dropout_rate 0.2 --activation_ch softmax --optimizer_ch sgd


import os,time
from ast import literal_eval
import argparse
import matplotlib.pyplot as plt 
import pandas as pd
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import scikitplot as skplt
import keras
from tensorflow.keras.utils import plot_model
from processData import dataset_selection
from networks import googlenet_layers
from networks import evaluate_test
from networks import select_optimiser

if not os.path.exists("model"):
    os.makedirs("model")

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
  help="dataset name, mnist or cifar10")
ap.add_argument("-m", "--model", required=True,
  help="model to train, vgg16 or googlenet")
ap.add_argument("-r", "--reshape", required=True, type=str, default="(224,224)",
  help="minimum reshape size for input image")
ap.add_argument("-b", "--batch_size", required=True, type=int,
  help="batch_size")
ap.add_argument("-e", "--epochs", required=True, type=int,
  help="number of epochs")
ap.add_argument("-l", "--learning_rate", required=True, type=float,
  help="learning rate")
ap.add_argument("-dr", "--dropout_rate", required=True, type=float,
  help="dropout_rate")
ap.add_argument("-a", "--activation_ch", required=True, 
  help="activation choice")
ap.add_argument("-o", "--optimizer_ch", required=True,
  help="optimizer choice")
args = vars(ap.parse_args())


model_output_csv = pd.DataFrame()

dataset_name = args['dataset']
model_name = args['model']
batch_size = args['batch_size']
buffer_size = 10000
epochs = args['epochs']
learning_rate = args['learning_rate']
dropout_rate = args['dropout_rate']
activation_ch = args['activation_ch']
opt = args['optimizer_ch']
num_classes = 10
input_dims = literal_eval(args['reshape'])




data = dataset_selection(dataset_name,final_shape=input_dims)
print("Train Dataset",data.x_train.shape,data.y_train.shape)
print("Validation Dataset",data.x_val.shape, data.y_val.shape)
print("Test Dataset",data.x_test.shape,data.y_test.shape)


for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.imshow(data.x_train[i], cmap="gray")
plt.show()
plt.savefig('data.png')


input_shape = data.x_train.shape[1:]
print("Creating network with Paramters")
print()
print("batch_size = ", batch_size)
print("buffer_size = ", buffer_size)
print("epochs = ", epochs)
print("learning_rate = ", learning_rate)
print("dropout_rate = ", dropout_rate)
print("activation_ch = ", activation_ch)
print("opt = ", opt)
print("num_classes = ", num_classes)
print("input_dims = ", str(input_shape))

if model_name == "vgg16":
	inputs = keras.layers.Input(shape=input_shape)
	model_output = vgg16_layers(inputs,num_classes,dropout_rate,activation_ch)

	opt_type = select_optimiser(opt,learning_rate)

	model = keras.models.Model(inputs=inputs, outputs=model_output)
	model.compile(loss=keras.losses.categorical_crossentropy,
	              optimizer=opt_type,
	              metrics=['accuracy'])
	plot_model(model, to_file='vgg16.png')

	print(model.summary())

	start = time.time()
	history = model.fit(data.x_train, data.y_train,
	                    batch_size=batch_size,
	                    epochs=epochs,
	                    verbose=1,
	                    validation_data=(data.x_val, data.y_val))
	end = time.time()
	total_exec_time = end - start
	print("Training Completed in ", str(total_exec_time/60), "mins")
	print("Saving Network and weights")
	model.save_weights('model/vgg16_wt.h5')

	print("Evaluating network performance on Test Data")
	score = model.evaluate(data.x_test, data.y_test, verbose=1,batch_size=batch_size)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])

	predictions = model.predict(data.x_test, verbose=0)

	# print(confusion_matrix(np.argmax(data.y_test,axis=1),np.argmax(predictions,axis=1), normalize='true'))
	skplt.metrics.plot_confusion_matrix(np.argmax(data.y_test,axis=1),np.argmax(predictions,axis=1), normalize=True,figsize = (7,7))
	print(classification_report(np.argmax(data.y_test,axis=1),np.argmax(predictions,axis=1)))

	fig = plt.figure(figsize=(12,5))
	subfig = fig.add_subplot(121)
	subfig.plot(history.history['accuracy'])
	subfig.plot(history.history['val_accuracy'])
	subfig.set_title('model accuracy')
	subfig.set_ylabel('accuracy')
	subfig.set_xlabel('epoch')
	subfig.legend(['train', 'val'], loc='upper left')
	subfig = fig.add_subplot(122)

	subfig.plot(history.history['loss'])
	subfig.plot(history.history['val_loss'])
	subfig.set_title('model loss')
	subfig.set_ylabel('loss')
	subfig.set_xlabel('epoch')
	subfig.legend(['train', 'val'], loc='upper left')
	plt.show()
	plt.savefig('vgg_acc_loss.png')

	p = evaluate_test(data.y_test,predictions)

	print("Error Analysis")
	print("True:          {}".format(np.argmax(data.y_test[p[0][0:5]], axis =1)))
	print("classified as: {}".format(np.argmax(predictions[p[0][0:5]], axis=1)))
	print("Error Cases")
	print("True:          {}".format(np.argmax(data.y_test[p[1][6:11]], axis =1)))
	print("classified as: {}".format(np.argmax(predictions[p[1][6:11]], axis=1)))

	print('Saving Network Results')
	hist = {}
	hist['model_name'] = model_name
	hist['dataset'] = dataset_name
	hist['dropout_rate'] = dropout_rate
	hist['activation'] = activation_ch
	hist['lr_rate'] = learning_rate
	hist['epochs'] = epochs
	hist['opt'] = opt
	hist['input_dims'] = data.x_train.shape[1:]
	hist = dict(hist, **history.history)
	hist['test_loss'] = score[0]
	hist['test_accuracy'] = score[1]
	np.save('model.txt',hist)
	model_output_csv = model_output_csv.append(hist, ignore_index=True)
	model_output_csv.to_csv("vgg_results.csv", index=False)

	print("Releasing GPU Memory")
	keras.backend.clear_session()

if model_name == 'googlenet':
	inputs = keras.layers.Input(shape=input_shape)
	(main, aux1, aux2) = googlenet_layers(inputs,num_classes,dropout_rate,activation_ch)

	opt_type = select_optimiser(opt,learning_rate)

	model = keras.models.Model(inputs=inputs, outputs=[main, aux1, aux2])
	model.compile(loss=keras.losses.categorical_crossentropy,
	              optimizer=opt_type,
	              metrics=['accuracy'])
	plot_model(model, to_file='inception.png')
	print(model.summary())

	start = time.time()
	history = model.fit(data.x_train, [data.y_train,data.y_train,data.y_train],
	                    batch_size=batch_size,
	                    epochs=epochs,
	                    verbose=1,
	                    validation_data=(data.x_val, [data.y_val,data.y_val,data.y_val]))
	end = time.time()
	total_exec_time = end - start
	print("Training Completed in ", str(total_exec_time/60), "mins")

	print("Saving Network and weights")
	weight_path = 'model/inception_wt.h5'
	model.save_weights(weight_path)
	print("Evaluating network performance on Test Data")
	score = model.evaluate(data.x_test, [data.y_test,data.y_test,data.y_test], verbose=1,batch_size=batch_size)
	print('Test loss:', score[0])
	print('Test accuracy:', score[4])
	predictions = model.predict(data.x_test, verbose=0)


	# print(confusion_matrix(np.argmax(data.y_test,axis=1),np.argmax(predictions[0],axis=1), normalize='true'))
	skplt.metrics.plot_confusion_matrix(np.argmax(data.y_test,axis=1),np.argmax(predictions[0],axis=1), normalize=True,figsize = (7,7))
	print(classification_report(np.argmax(data.y_test,axis=1),np.argmax(predictions[0],axis=1)))

	fig = plt.figure(figsize=(12,5))
	subfig = fig.add_subplot(121)
	subfig.plot(history.history['main_accuracy'])
	subfig.plot(history.history['val_main_accuracy'])
	subfig.plot(history.history['aux1_accuracy'])
	subfig.plot(history.history['val_aux1_accuracy'])
	subfig.plot(history.history['aux2_accuracy'])
	subfig.plot(history.history['val_aux2_accuracy'])
	subfig.set_title('model accuracy')
	subfig.set_ylabel('accuracy')
	subfig.set_xlabel('epoch')
	subfig.legend(['main_accuracy', 'val_main_accuracy','aux1_accuracy','val_aux1_accuracy','aux2_accuracy','val_aux2_accuracy'], loc='upper left')
	subfig = fig.add_subplot(122)
	# summarize history for loss
	subfig.plot(history.history['main_loss'])
	subfig.plot(history.history['val_main_loss'])
	subfig.plot(history.history['aux1_loss'])
	subfig.plot(history.history['val_aux1_loss'])
	subfig.plot(history.history['aux2_loss'])
	subfig.plot(history.history['val_aux2_loss'])
	subfig.set_title('model loss')
	subfig.set_ylabel('loss')
	subfig.set_xlabel('epoch')
	subfig.legend(['main_loss', 'val_main_loss','aux1_loss','val_aux1_loss','aux2_loss','val_aux2_loss'], loc='upper left')
	plt.show()
	plt.savefig('googlenet_acc_loss.png')

	# loss
	fig = plt.figure(figsize=(12,5))
	subfig = fig.add_subplot(121)
	subfig.plot(history.history['loss'])
	subfig.plot(history.history['val_loss'])
	subfig.set_title('model loss')
	subfig.set_ylabel('loss')
	subfig.set_xlabel('epoch')
	subfig.legend(['train', 'val'], loc='upper left')
	plt.show()
	plt.savefig('googlenet_loss.png')


	p = evaluate_test(data.y_test,predictions[0])

	print("True:          {}".format(np.argmax(data.y_test[p[0][0:5]], axis =1)))
	print("classified as: {}".format(np.argmax(predictions[0][p[0][0:5]], axis=1)))
	print("Error Cases")
	print("True:          {}".format(np.argmax(data.y_test[p[1][6:11]], axis =1)))
	print("classified as: {}".format(np.argmax(predictions[0][p[1][6:11]], axis=1)))
	
	print('Saving Network Results')
	hist = {}
	hist['model_name'] = 'inception'
	hist['dataset'] = dataset_name
	hist['dropout_rate'] = dropout_rate
	hist['activation'] = activation_ch
	hist['lr_rate'] = learning_rate
	hist['epochs'] = epochs
	hist['opt'] = opt
	hist['input_dims'] = data.x_train.shape[1:]
	hist = dict(hist, **history.history)
	hist['test_loss'] = score[0]
	hist['test_accuracy'] = score[4]
	np.save('model.txt',hist)
	model_output_csv = model_output_csv.append(hist, ignore_index=True)
	model_output_csv.to_csv("inception_results.csv", index=False)
	print("Releasing GPU Memory")
	keras.backend.clear_session()
