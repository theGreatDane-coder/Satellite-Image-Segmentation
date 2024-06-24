# Satellite Image Land Cover Segmentation using U-net 


## Files Explanation
In this section we will present the different files inside the repository as well as an explanation about their functionality


|File Name| Explanation / Function |
|---------|------------|
|`train.py` | Used for the training of the network.  |
|`predict.py`|Used to perform a prediction on the test dataset. |
|`eval.py`|Used to evaluate the mdoel using cross-entropy loss (CE). |
|`classcount.py`| Calculates the weights to be used for the weighted cross entropy by counting the pixel number of each class in the train dataset.|
|`distribution.py`| Used to evaluate the pixel numbers of the validation and training set and visualize them via  bar chart.|
|`dataset.py`| Used as a data loader during the training phase.|
|`data_vis.py`| Helper to visualize the data.|

## Training

The following flags can be used while training the model.

<ins>_Guidelines_<ins>

`-f` : Used to load a model already stored in memory. \
`-e` : Used to specify the Number of training epochs. \
`-l` : Used to specify the learning rate to be used for training. \
`-b` : Used to specify the batch size. \
`-v` : Used to specify the percentage of the validation split (1-100). \
`-s` : Used to specify the scale of the image to be used for training.

<ins>_Example:_<ins/>

Training the model for 100 epochs using 20% of the data as a validation split, learning rate is 4x10^-5, batch size is 2 and image scale is 20%

`!python3 train.py -e 100 -v 20.0 -l 4e-5 -b 2 -s 0.2`

## Prediction
<ins>_Guidelines_<ins>

`-m` : Used to specify the directory to the model. \
`-i` : Used to specify the directory of the images to test the model on. \
`-o` : Used to specify the directory in which the predictions will be outputted. \
`-s` : Used to specify the scale of the images to be used for predictions. \
`--viz:` Saves the predictions in the form of an image. \
(For best results used the same scale you used for training the model)

_Note:_ Inference of scale 0.2 takes approximately 10 minutes.

<ins>_Example_<ins>

Making a prediction on the full test set dataset using 30 epoch model trained on full data using a scale of 20%. The script  outputs the IoU score of the model.

```
%%time
!python predict.py -m data/checkpoints/model_ep30_full_data.pth -i data/<test_set_directory>/* -o predictions/ -s 0.2 --viz
```