# pyqt-torch-text-classification
<div align="center">
  <img src="https://user-images.githubusercontent.com/55078043/229002952-9afe57de-b0b6-400f-9628-b8e0044d3f7b.png" width="150px" height="150px"><br/><br/>
  
  [![](https://dcbadge.vercel.app/api/server/cHekprskVE)](https://discord.gg/cHekprskVE)

  [![](https://img.shields.io/badge/한국어-readme-green)](https://github.com/yjg30737/pyqt-torch-text-classification/blob/main/README.kr.md)
</div>

Example of using text classification from texts with pytorch model in Python PyQt GUI

This small program uses an AI model to determine whether the text you input is positive or negative. It assesses where your entered text falls in terms of sentiment.

The text classification model used in this program has been trained 20 times (epochs) on the following data.

```
texts = ["It's so fun.", "It's the best.", "It's a really well-made movie.", 
         "It's a movie I want to recommend.", "I want to see it again.", 
         "Well, I'm not sure.", "It's not very good.", 
         "It's more boring than I thought.", "The acting is awkward.", 
         "It's not fun."]
```

The source code is available in a <a href="https://www.kaggle.com/code/yoonjunggyu/pytorch-text-classification">Kaggle notebook</a>.

Text strings cannot be directly input as data for model training, so they must be converted into tensors. For this conversion, methods such as **one-hot encoding** or **word embedding** are used. **Word embedding** is more effective than one-hot encoding as it is more memory-efficient and captures the context between words.

Since the dataset is very small, the model is also quite small! Due to the small size of the model, its accuracy on new data is significantly reduced. However, this is not focused on creating an accurate text classification model, but rather an example of how to make basic text classification model and apply the model in a GUI.

Please refer to the source in the Kaggle notebook and ask GPT for an interpretation of each source. I will respond kindly :)

If you want to do the text classification from number-based dataset, <a href="https://github.com/yjg30737/pyqt-torch-text-classification-from-number">try this</a>.

## Requirements
* PyQt5 >= 5.14
* torch

## How to Run
1. git clone ~
3. pip install -r requirements.txt
4. python run ~

## Preview
![image](https://github.com/yjg30737/pyqt-torch-text-classification/assets/55078043/4b3941e3-9a51-4b62-9a05-ca0bc929a936)
