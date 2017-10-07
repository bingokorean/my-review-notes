# Linguistic Regularities in Continuous Space Word Representations (2013)

A paper note written in korean is in [**here**](https://1drv.ms/p/s!AllPqyV9kKUrhBg0yG4S2XFXNNxz)

## Summary
* (FNN, RNN based) Language Model을 학습하면 distributed word representation을 추출할 수 있다. 구체적으로는 RNNLM에서 input과 hidden을 연결하는 weight matrix 그 자체를 continuous word representation으로 사용한다.
* 신기하게도 이러한 word representation에서 syntactic과 semantic regularities를 추출할 수 있었다.
* word representation이 vector이기 때문에 덧셈, 뺄셈과 같은 벡터 연산과 벡터간의 similairty를 측정할 수 있는 cosine distance를 사용하여 평가를 실시한다. 다시 말해서, analogy question을 정의하고 vector offset method를 사용해서 실험을 한다. 
 
