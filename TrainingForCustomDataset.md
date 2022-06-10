# Training SparseInst For Custom dataset
I tried to use SparseInst to train for my dataset, but the [README](./README.md) of SparseInst and the [document](https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html) of [detectron2](https://github.com/facebookresearch/detectron2) are difficult to understand. However, I found method to use custom dataset to train SparseInst. I want to write something to record the training steps.

## Prepare Your Dataset
If you want to training your SparseInst, you must prepare your dataset first. The format of the dataset must follow COCO.

For example, I want to detect animals that include dog, cat and bird. First, I collect training images, and then annotating them. The structure of the dataset will be:
```
|--- animal_dataset/
	|--- annotations/
 	|	|--- instances_train_animals.json
	|	|--- instances_val_animals.json
	|
	|--- images/
		|--- train/
		|--- val/
```
Here, I use `animal_dataset` as the root directory name. And then the json files `instances_train_animals.json` and `instances_val_animals.json` are annotations for training and validation datasets. The directory `images` will contain all images for training and validation datasets. Note, the format of the json files must follow COCO.

## Register Custom Dataset
You must write code to resgster your dataset. The API detectron2 can use your code to obtain the dataset.

I give a example for demonstraction. I use `animal_dataset` as my dataset name. And then I edit the file `train_net.py` to register my dataset. Here is my code:
```python
from detectron2.data.datasets import register_coco_instances
register_coco_instances("animals_dataset_train", {}, "./datasets/animal_dataset/annotations/instances_train_animals.json", "./datasets/animal_dataset/images/train")
register_coco_instances("animals_dataset_val", {}, "./datasets/animal_dataset/annotations/instances_val_animals.json", "./datasets/animal_dataset/images/val")
```
The first line is to import the function `register_coco_instances()`. The second line uses `animals_dataset_train` to register training dataset, and then the third line uses `animals_dataset_val` to register the validation dataset. The function `register_coco_instances()` need three paramenters:
- The dataset name
- A dict that has no element
- The json file for the dataset
- The images path

## Set The List Of Category Names
Next, the API detectron2 need to know the list of the category names for the custom dataset, so we must write code in the file `train_net.py` to do it.

Here is the exmaple to set the list of the category names for my dataset `animal_dataset`:
```python
from detectron2.data import MetadataCatalog
MetadataCatalog.get("animals_dataset_train").thing_classes = ["dog", "cat", "bird"]
MetadataCatalog.get("animals_dataset_val").thing_classes = ["dog", "cat", "bird"]
```
The first line is to import the class `MetadataCatalog`. We can this class to set the list of the category names. Note, you must use your dataset name to set the list.

## Edit yaml File To Load Custom Dataset
This is final step. You can find many yaml files in the directory `configs/`. The file `configs/Base-SparseInst.yaml` is the base yaml file. Other yaml files use it to set configuration. You need edit one of the yaml files, or use one of them as base configuration to train your model.

For example, I use `configs/sparse_inst_r50_base.yaml` as base configuration to train my model. I copy it as a transcript, and then edit it to set configuration. Here is my yaml file:
```yaml
_BASE_: "Base-SparseInst.yaml"
MODEL:
  SPARSE_INST:
    DECODER:
      NAME: "BaseIAMDecoder"
DATASETS:
  TRAIN: ("animals_dataset_train",)
  TEST:  ("animals_dataset_val",)
SOLVER:
  IMS_PER_BATCH: 8
  BASE_LR: 0.0005
  STEPS: (167090, 199640)
  MAX_ITER: 217000
  WEIGHT_DECAY: 0.05
OUTPUT_DIR: "output/sparse_inst_r50_base"
```
The node `DATASETS` has two sub node `TRAIN` and `TEST`. In my yaml file, I set the value of the sub node `TRAIN` to `("animals_dataset_train",)`. I use `("animals_dataset_val",)` as the value for the sub node `TEST`. We must change the values of the sub node to load the custom dataset. Note, you must follow the format `(<dataset_name> ,)` to set dataset, where `<dataset_name>` is your dataset name.

Finally, you can use command to train your model. Good luck!
