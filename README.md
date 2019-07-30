## Keras Implement of Lazy Optimizer

Inheriting Optimizer class, wrapping the original optimizer to achieve a new corresponding lazy optimizer.

Here we use gradients are equal to zeros or not to distinguish whether the words are sampled or not.

### Usage 
just replace your original with-momentum optimizer, like `Adam(1e-3)`, with `LazyOptimizer(Adam(1e-3), embedding_layers)`.

see <a href="https://github.com/bojone/keras_lazyoptimizer/blob/master/imdb_lstm_test.py">imdb_lstm_test.py</a>.

## Lazy类优化器的Keras实现

继承Optimizer类，包装原有优化器，实现Lazy版优化器。

这里判断一个词是否被采样的方法是检查该词的梯度是否全为0。

### 用法
直接将原来用的带动量的优化器, 如 `Adam(1e-3)`, 替换为 `LazyOptimizer(Adam(1e-3), embedding_layers)` 就行了.

参考 <a href="https://github.com/bojone/keras_lazyoptimizer/blob/master/imdb_lstm_test.py">imdb_lstm_test.py</a>.
