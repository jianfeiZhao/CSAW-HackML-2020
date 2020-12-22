import sys

from PIL import Image
import numpy as np
from keras.models import load_model

from eval import *

model_dict = {"sunglasses": ["models/sunglasses_bd_net.h5",
                             "repaired_models/sunglasses_bd_net_repaired.h5"],
              "multi_trigger_multi_target": ["models/multi_trigger_multi_target_bd_net.h5",
                                             "repaired_models/multi_trigger_multi_target_bd_net_repaired.h5"],
              "anonymous_2": ["models/anonymous_2_bd_net.h5",
                              "repaired_models/anonymous_2_bd_net_repaired.h5"],
              "anonymous_1": ["models/anonymous_1_bd_net.h5",
                              "repaired_models/anonymous_1_bd_net_repaired.h5"]}

model_name = str(sys.argv[1])
filename = str(sys.argv[2])

if model_name not in model_dict.keys():
    print("Model should be choose from ", list(model_dict.keys()))
    sys.exit(0)

repaired = load_model(model_dict[model_name][0])
origin = load_model(model_dict[model_name][1])

if filename.split(".")[-1] == 'h5':
    x_test, y_test = data_loader(filename)
    x_test = x_test / 255
    pred_retrain = repaired.predict(x_test)
    pred_origin = origin.predict(x_test)
    label_retrain = np.argmax(pred_retrain, axis=1)
    label_origin = np.argmax(pred_origin, axis=1)

    cnt = 0
    for i in range(len(y_test)):
        if not (label_retrain[i] == label_origin[i] and
                pred_origin[i][label_retrain[i]] - pred_retrain[i][label_retrain[i]] <= 0.5):
            label_retrain[i] = 1283
            cnt += 1

    accu = np.mean(np.equal(label_retrain, y_test)) * 100
    print('\n')
    print('accuracy after repairing: ', accu)
    print('num of trojan input: ', cnt)

else:
    image = np.expand_dims(np.asarray(Image.open(filename)), axis=0)
    pred_retrain = repaired.predict(image)
    pred_origin = origin.predict(image)
    label_retrain = np.argmax(pred_retrain)
    label_origin = np.argmax(pred_origin)

    if not (label_retrain == label_origin and
            pred_origin[label_retrain] - pred_retrain[label_retrain] <= 0.5):
        print('\n')
        print(1283)
    else:
        print('\n')
        print(label_retrain)
