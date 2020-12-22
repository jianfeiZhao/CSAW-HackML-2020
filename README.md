# CSAW-HackML-2020

## Repairing the Model
Execute `repair.py` to repair the model. 
Pre-repaired models are saved in `repaired_models/` 
using given validation set for re-training.

## Evaluating Performance
1. To evaluate the repairing performance, 
execute `eval_repaired.py` by running:  
`python3 eval_repaired.py <model name> <data directory>`  
where model name is chosen from `[sunglasses, multi_trigger_multi_target, anonymous_2, anonymous_1]`, 
and data is either a image file or a `.h5` dataset.  
E.g. `python3 eval_repaired.py multi_trigger_multi_target data/demo.jpeg`  
and `python3 eval_repaired.py sunglasses data/data.h5`
2. If a image file is given, the script will output the label predicted (1283 for poisoned images, \[0, 1282\] for clean images). 
If a dataset is given, the script will output accuracy of repaired model's prediction, 
and number of poisoned images detected in the dataset. 
Note that the input dataset should have poisoned images labeled as class_1283 so that the accuracy can be reported correctly.
3. A demo dataset which contains both poisoned and clean images with correct labels can be found [here](https://drive.google.com/file/d/1aQ818PSyGvri3hWl749876VeWgaN8IK7/view?usp=sharing), 
and save it under `data/` folder.

