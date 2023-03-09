# ESNet
ESNet is a machine learning model based on convolutional neural network and convolutional block attention modules (CBAM) 
for the prediction of formation energy of crystal structure. 
It uses **density of states (DOS) of unrelaxed initial** structures as input representations, 
achieving successful performance in IS2RE task. 
![model_architecture](https://user-images.githubusercontent.com/80326874/224038629-99d5dd05-66b7-4873-9953-67c4e2a225e5.png)

## Installation
Run the following command to set up
```
conda update conda
conda create -n ESNet python=3.8
pip install tensorflow-gpu==2.4.1
```

## Exmaple Use
### Input files generation
To prepare the input of the model, you will need to [download all data](#data) that we provide. Then, please run:
```
python preprocessing/DOS_preprocess.py
```
### Train ESNet
Hyperparameters, such as `--batch_size`, `--init_lr`, `--lr_sch_factor`, `--lr_sch_patience`, `--loss`, `--epochs` can be set. Their default values are set as 64, 0.0015, 0.5, 15, 'logcosh', and 300, respectively, now.
```
python train.py
```
### Evaluate saved ESNet
Before evaluate the fully-trained model, [preprocessed data for evaluation](#data) should be downloaded first. These are 2e-ORR dataset for both IS2RE and RS2RE tasks:
* evaluate/2eORR_init_dos.pkl, evaluate/2eORR_init_target.pkl
* evaluate/2eORR_relax_dos.pkl, evaluate/2eORR_relax_target.pkl

Then, you can evaluate the provided ESNet, specifying the evaluation task through `--evaluation_version`.
```
python evaluate.py --evaluate_version='IS2RE'
```
The predicted formation energy will be in the units of `eV/atoms`

## Data
All data used during the training and validation can be downloaded from the link below.

<https://ewhainnet-my.sharepoint.com/:f:/g/personal/222hgg15_ewhain_net/Eumo24XnIoJHvQSaaLncjYEB1xsiLuRYFlMl_1-FugfOng?e=CQf8cm>

Here, we also provide initial and relaxed crystal structures of the 2e-ORR dataset, used as validation sets in this work. These can be utilized to evaluate other baselines that do not use DOS as input, such as CGCNN, MEGNet, Wren, and Roost.
* all/2eORR_init_data.7z
* all/2eORR_relax_data.7z
