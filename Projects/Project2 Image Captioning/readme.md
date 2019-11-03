## Image Captioning Project

### Instructions  
1. Clone this repo: https://github.com/cocodataset/cocoapi  
```
git clone https://github.com/cocodataset/cocoapi.git  
```

2. Setup the coco API (also described in the readme [here](https://github.com/cocodataset/cocoapi)) 
```
cd cocoapi/PythonAPI  
make  
cd ..
```
(Did not perform this step in google colab)

3. Download some specific data from here: http://cocodataset.org/#download (described below)

* Under **Annotations**, download:
  * **2014 Train/Val annotations [241MB]** (extract captions_train2014.json and captions_val2014.json, and place at locations cocoapi/annotations/captions_train2014.json and cocoapi/annotations/captions_val2014.json, respectively)  
  * **2014 Testing Image info [1MB]** (extract image_info_test2014.json and place at location cocoapi/annotations/image_info_test2014.json)

* Under **Images**, download:
  * **2014 Train images [83K/13GB]** (extract the train2014 folder and place at location cocoapi/images/train2014/)
  * **2014 Val images [41K/6GB]** (extract the val2014 folder and place at location cocoapi/images/val2014/)
  * **2014 Test images [41K/6GB]** (extract the test2014 folder and place at location cocoapi/images/test2014/)

4. The project is structured as a series of Jupyter notebooks that are designed to be completed in sequential order (`0_Dataset.ipynb, 1_Preliminaries.ipynb, 2_Training.ipynb, 3_Inference.ipynb`).

### Model Architecture:
1. **Encoder Model:** A pretrained resnet with a new embedding layer. The last layer(new layer) transforms the CNN feature vector to a specified embedding size(embedding size of text).
2. **Decoder Model:** An Embedding layer, followed by an LSTM layer and a  fully connected layer. LSTM layer takes in feature vector from encoder along with captions passed through embedding layer. Then the fully connected layer tranforms the ouput of the LSTM to a probability distribution among the vocabulary set, to predict the next word.

#### Files:
- 2 epoch folder contains files run when 2/1 epochs were completed and its inferences. 
- 3 epoch folder contains files run when an additional was completed.
- 0,1 ipynb are initial exploratory files
- model.py contains the model architecture
- vocab.pkl is the generated vocabulary file
- **All files were run in google colab**
- zip file contains project files from udacitys workspace. These files contain the retrained model, the answered questions in the notebooks and slight change in the `clean` function in notebook_3
