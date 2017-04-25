# Transfer Learning Tutorial

A guide to train the inception-resnet-v2 model in TensorFlow. Visit [here](https://kwotsin.github.io/tech/2017/02/11/transfer-learning.html) for more information.

**Update**: Note that if you're using the old TF 0.12 code that uses `loss = slim.losses.softmax_cross_entropy(predictions, one_hot_labels)`, and decide to update to using the `tf.losses.softmax_cross_entropy` function,you should also change the positions of the arguments. For instance, you should do this:  `tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=predictions)`. 

If you keep `predictions` in the first argument and `one_hot_labels` in the second argument, you will encounter a problematic loss function that doesn't really help to train your model (as some of you have emailed me). I have updated the code to correct this issue that may be hard to detect (because the model will still train, except it trains poorly).
