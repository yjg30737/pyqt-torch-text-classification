한국어 설명:

이 작은 프로그램은 텍스트의 긍정적임, 부정적임 여부를 판단하는 ai 모델을 사용하여 당신이 입력한 텍스트가 어디에 속하는지에 대한 여부를 판단합니다.

이 작은 프로그램에서 사용된 텍스트 분류 모델은 다음 데이터를 20번 (epochs) 훈련시켰습니다. kaggle notebook에 소스가 존재합니다.

이 예제에서 사용되는 모델은 다음 데이터를 훈련시켜서 새로운 데이터에 대한 긍정, 부정 여부 예측을 실시합니다.

```
texts = ["It's so fun.", "It's the best.", "It's a really well-made movie.", 
         "It's a movie I want to recommend.", "I want to see it again.", 
         "Well, I'm not sure.", "It's not very good.", 
         "It's more boring than I thought.", "The acting is awkward.", 
         "It's not fun."]
 ```

텍스트 스트링은 모델 훈련 시 데이터로 곧바로 입력할 수 없기 때문에 텐서로 변환해야 합니다. 이 때 제로 핫 인코딩 혹은 워드 임베딩이라는 방법이 사용되는데, 워드 임베딩은 제로 핫 인코딩에 비해 메모리 절약적이며 단어 간 문맥을 파악하기 때문에 더욱 효과적인 방법입니다.

데이터세트가 매우 작기 때문에 모델도 매우 작습니다! 모델이 작기 때문에 새로운 데이터에 대한 정확도도 크게 떨어집니다. 하지만 이것은 정확한 텍스트 분류 모델에 초점을 맞춘 것이 아닌, 어떤 식으로 GUI에 텍스트 모델을 적용할 수 있는가에 대한 예제입니다.

kaggle notebook에 있는 소스를 참고하여, gpt에 각 소스에 대한 해석을 부탁해 보세요 :) 친절하게 대답해줄 것입니다.

설치 방법과 미리보기는 README.md를 참고해주세요 !
