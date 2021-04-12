# mosaic

## Authors
Audrea Huang \<audreahuang@gmail.com> <br>
Péter Hámori \<hampet97@gmail.com>

## Description
Correctly identifying objects portrayed in images remains a significant challenge in computer vision, where existing models are trained on dominant figures and identities within a contemporary context. Data bias of this form neglects underrepresented communities and variations of objects, thereby limiting extensions to more specific topics such as census reporting or early modern artwork. This project aims to fine tune models to recognize various types of objects by providing a wider source of training images and analyzing patterns within them. The results of this work can be applied to support image editing software, aid in research methods for digital humanities, and increase the efficiency of conducting inventory.

## Data Loading
Download 2014 Train/Val annotations from https://cocodataset.org/#download and name the folder "annotations_trainval2014"<br>
Run `coco_data_load.ipynb`

## Training and Evaluation
Store your path to `instances_train2014.json` in `annFileTrain`<br>
Run `coco_load_train_eval.ipynb`

