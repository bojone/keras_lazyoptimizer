## Keras Implement of Lazy Optimizer

Inheriting Optimizer class, wrapping the original optimizer to achieve a new corresponding lazy optimizer.
Here we use gradients are equal to zeros or not to distinguish whether the words are sampled or not.

Usage: just replace `Adam(1e-3)` with `LazyOptimizer(Adam(1e-3), embedding_layers)`, see <a href="https://github.com/bojone/keras_lazyoptimizer/blob/master/imdb_lstm_test.py">imdb_lstm_test.py</a>.
