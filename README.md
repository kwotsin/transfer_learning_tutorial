# Transfer Learning Tutorial

A guide to train the inception-resnet-v2 model in TensorFlow. Visit [here](https://kwotsin.github.io/tech/2017/02/11/transfer-learning.html) for more information.

### FAQ:

**Q:** Why does my evaluation code give such a poor performance although my training seem to be fine?

**A:** This could be due to an issue of how `batch_norm` is updated during training in the newer versions of TF, although I've not have the chance to investigate this properly. However, some users have mentioned that by setting `is_training=True` back in the eval code, the model works exactly as expected. You should try this method and see if it works for you.

For more information, please see this thread: https://github.com/kwotsin/transfer_learning_tutorial/issues/11

**Q:** How do I only choose to fine-tune certain layers instead of all the layers?

**A:** By default, if you did not specify an argument for `variables_to_train` in the function `create_train_op` (as seen in the `train_flowers.py` file), this argument is set to `None` and will train all the layers instead. If you want to fine-tune only certain layers, you have to pass a list of variable names to the `variables_to_train` argument. But you may ask, "how do I know the variable names of the model?" One simple way is to simply run this code within the graph context:

```
with tf.Graph().as_default() as graph:
    .... #after you have constructed the model in the graph etc..
    for i in tf.trainable_variables():
        print i
```

You will see the exact variable names that you can choose to fine-tune.

For more information, you should visit the [documentation](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/learning.py#L374).

---

**Q:** Why is my code trying to restore variables like `InceptionResnetV2/Repeat_1/block17_20/Conv2d_1x1/weights/Adam_1` when they are not found in the .ckpt file?

**A:** The code is no longer trying to restore variables from the .ckpt file, but rather, from the log directory where the checkpoint models of your previous training are stored. This error happens when you have changed the code but did not remove the previous log directory, and so the Supervisor will attempt to restore a checkpoint from your previous training, which will result in a mismatch of variables. 

**Solution: Simply remove your previous log directory and run the code again.** This applies to both your training file and your evaluation file. See this [issue](https://github.com/kwotsin/transfer_learning_tutorial/issues/2) for more information.

---

**Q:** Why is my loss performing so poorly after I updated the loss function from `slim.losses.softmax_cross_entropy` to `tf.losses.softmax_cross_entropy`?

**A:** The position of the arguments for the one-hot-labels and the predictions have changed, resulting in the wrong loss computed. This happens if you're using an older version of the repo, but I have since updated the losses to `tf.losses` and accounted for the change in argument positions.

**Solution: `git pull` the master branch of the repository to get the updates.**

---

**Q:** Why does the evaluation code fails to restore the checkpoint variables I had trained and saved? My training works correctly but the evaluation code crashes.

**A:** There was an error in the code that mistakenly allows the saver used to restore the variables to save the model variables after the training is completed. Because we made this saver exclude some variables to be restored earlier on, these excluded variables will not be saved by this saver if we use it to save all the variables when the training to be completed. Instead, the code should have used the Supervisor's saver that exists internally to save the model variables in the end, since all trained variables will then be saved.

Usually, this does not occur if you have trained your model for more than 10 minutes, since the Supervisor's saver will save the variables every 10 minutes. However, if you end your training before 10 minutes, the wrong saver would have saved only some trained variables, and not all trained variables (which is what we want).

**Solution: `git pull` the master branch of the repository to get the updates.** I have changed the training code to make the supervisor save the variables at the end of the training instead. 
