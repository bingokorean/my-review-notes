# Linguistic Regularities in Continuous Space Word Representations (2013)

A paper note (written in korean) is in [**HERE (.ppt)**](https://1drv.ms/p/s!AllPqyV9kKUrhBg0yG4S2XFXNNxz) 

## Summary
* (FNN, RNN based) Neural Network Language Model을 학습하면 distributed word representation을 추출할 수 있음. (ex.RNNLM에서 input과 hidden을 연결하는 weight matrix 그 자체를 continuous word representation으로 사용)
* 신기하게도 이러한 word representation에서 syntactic과 semantic regularities를 발견할 수 있음.
* word representation이 continuous space vector이기 때문에 덧셈, 뺄셈과 같은 벡터 연산과 벡터간의 similairty를 측정할 수 있는 cosine distance를 사용하여 syntactic/semantic regularities가 있는지 확인. 다시 말해서, analogy question을 정의하고 vector offset method를 사용해서 확인. (ex. Vector(King) - Vector(Man) + Vector(Woman) = Vector(Queen))
 
