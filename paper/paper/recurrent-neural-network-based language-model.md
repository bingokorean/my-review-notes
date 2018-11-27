# [Recurrent neural network based language model](http://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf)

## Review
Review-note written by both korean and english is [**here**](https://1drv.ms/w/s!AllPqyV9kKUrgiNcJ6evHVW1AH9e).


## Summary

Summary involves not only the content of paper but also my subjective notion (e.g, especially, why?). Instead, Main Point and Keyword are totally based on the content of paper. My reading process for understanding is as follows: First, I have several intensive reading for getting 70% understanding. Second, I try to get 80% understanding while translating it into Korean. Third, I write Main Point and Keyword for repetition. Finally, I write summary with my subjective notion for getting 85% understanding. To reach over 90% understanding, I need to combine all the knowledge related to this fields. Warning: since this learning process is just my goal, I may misunderstand some concepts. 

### Keyword
* Sequential data prediction
* Statistical language modeling
* Advanced n-gram based model (cache-based, class-based)
* Recurrent neural network
* Feed-forward neural network
* Dynamic model (online learning)
* Back-propagation through time
* Perplexity reduction (WER reduction)

### Main Point
* About recurrent neural network for language modeling
* Language modeling is implemented by simple RNNs called Elman networks, where the direction of recurrent is not from context(t-1) to context(t), but to input(t).
* Simple RNNs are easy to implement and train. (simple RNNs has one less number of weight than original RNNs.)
* Beating both feed-forward neural network (which needs fixed-length input) and advanced n-gram statistic model (cache-based, class-based) via speech recognition experiments.
* Show that RNNs work well even smaller data compared to other n-gram based models.
* RNNs need very long training time as data gets bigger.
* FNNs need fixed-length input. Instead, RNNs can get arbitrary-length input. RNNs look more general model than FNNs in terms of the length of input.
* FNNs are trained by standard backpropagation, which updates the model at each time. Instead, RNNs are trained by backpropagation though time.
* Dynamic model (online learning) is good for Language modeling.
* Simple or standard RNNs suffer from long-term dependency problem.
* Similar to n-gram models, RNN LM does not have specific assumptions. The input of RNN LM is just 1-of-N encoding. So, they are easily adapted to new domain.

### Sum-up
In Artificial Intelligence and Machine Learning, sequential data prediction is considered as difficult problem because the prediction is not just a single number or integer number for classification problem, but like sentence, image for generating problem.

Even though n-gram based models have weak linguistic assumption (e.g., language consists of atomic symbols or words), those kinds of models (cache-based, class-based n-gram language model) are state-of-the-art for language modeling at that time (2010 when the paper is written). That means n-gram-based models work better than models which have strong linguistic assumption such as parse tree, morphology (e.g., syntax, semantic). I think, co-occurrence is important for language modeling. However, in practice those n-gram-based models do not work well and they work in specific dataset or domain.

Recently, feed-forward neural network models are emerging. They could defeat one of state-of-the-art models like class-based n-gram model via speech recognition experiments. However, a major deficiency of such feed-forward NNs is that they need fixed-length input. It looks some limitations in that the length of useful context can vary according to the length of sentence. If there is a long sentence, long context would be needed.

Therefore, the goal of this research is to defeat both state-of-the-art such as n-gram-based models and feed-forward neural networks and to improve the deficiencies of those two models by using simple recurrent neural models.

Since it is a kind of neural network model, it has large model complexity, which has advantage of memorizing well as bigger data is coming. Surprisingly, experiments in this paper show that RNNs work well even though they are trained with even smaller amount of data than other models. I think it is because of neural network model itself and I believe that the more data, the even better model in case of neural network models compared to other models. Also, Since in recurrent neural networks, information can cycle inside these networks for arbitrarily long time, we do not need fixed-length input for those models. It leads for the model to memorize context well even long sentence is coming. But, recent research mention that simple RNN is hard to be trained when there is long-term dependency. (So, now we have LSTM.)

Interesting thing is that simple version of RNN called Elman network in this paper is a little different from standard RNN. In Elman network, the direction of recurrent is not from context(t-1) to context(t), but to input(t). It leads for the model to have one less weight in the model. So, it is relatively easy to train and also implement.

For evaluating the model, two kinds of speech recognition experiments were conducted. One is standard test and another one is more reliable test, where test data was on independent headset condition. From both experiments, RNN has nice perplexity reduction. The author comments that when using RNN LM, dynamic model (online learning) is useful and backpropagation though the time is also useful.

As a conclusion, I think this paper is a foundation for Language modeling, Machine translation where recurrent neural networks are actively used these days. After reading this paper, I could have some insight about potential of recurrent neural networks.


### Reference
Mikolov, Tomas, et al. "Recurrent neural network based language model." Interspeech. Vol. 2. 2010.
