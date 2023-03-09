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

## Data
All data used during the training and validation can be downloaded from the link below.
<>
