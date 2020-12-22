import os

import numpy as np
from keras.models import load_model, Model

from eval import *


x_train, y_train = data_loader("data/clean_validation_data.h5")
x_train = x_train / 255
model_names = [model_name for model_name in os.listdir('models/') 
			   if model_name.split('.')[0][-3:] == 'net']
for model_name in model_names:
	model = load_model(os.path.join('models', model_name))
	conv_3_model = Model(inputs=model.input,
                     outputs=model.get_layer("pool_3").output)
	conv_3_clean_output = np.mean(conv_3_model.predict(x_train), axis=0)
	idx = np.argsort(np.sum(conv_3_clean_output, axis=(0, 1)))
	layer = model.get_layer("conv_3")
	weight, bias = layer.get_weights()
	pred_clean = np.argmax(model.predict(x_train), axis=1)
	accu_clean = np.mean(np.equal(pred_clean, y_train)) * 100
	i = 0
	while accu_clean >= 50:
	    cur_idx = idx[i]
	    weight[:, :, :, cur_idx] = 0
	    bias[cur_idx] = 0
	    layer.set_weights([weight, bias])
	    pred_clean = np.argmax(model.predict(x_train), axis=1)
	    accu_clean = np.mean(np.equal(pred_clean, y_train))*100
	    i += 1

	model.fit(x=x_train, y=y_train, batch_size=32, epochs=20, verbose=0)
	save_name = model_name.split('.')[0] + "_repaired.h5"
	model.save(os.path.join('repaired_models/', save_name))
	print(model_name + " git srepaired")

