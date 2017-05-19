# Readme !


---
### Author : Etienne HOUZE
#### Date : April-July 2017
#### Bentley Internship - Acute3D
---
## Context and goal
This repository is a project realized with the Keras API during an internship in Bentley Systems.

It is meant to study and implement neural network based image dense labelling with further goal to label 3D models. Since labelled 3D meshes data is not easily found, it was preferable to study labelling on the input 2D images.

The dataset used in this project is CityScape dataset, see the references at the end of this file.

## Repo organization
### Requirements :
This project requires keras, and is built on its tensorflow backend. Make sure to have those packages installed, as well as an up-to-date numpy, matplotlib, pillow and pydot packages.

### Architecture
#### DistributedLearning.py
The main script of the project. Basically, it is just a sandbox script to test stuff and launch runs

#### src
A folder containg source scripts describing layers, callback functions and models.

#### helpers
A folder containg scripts for preprocessing data, generating batches and the label-defining file, taken from CityScape dataset

## References

Keras API : https://github.com/fchollet/keras

CityScape dataset : https://www.cityscapes-dataset.com/