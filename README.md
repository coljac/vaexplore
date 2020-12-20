# Vaexplore

Plotly dash app to explore the results of generative deep learning experiments.

First commit assumes 1D data.

To use:

```
from vaeexplore import Model

def encode(examples):
    """ Take test examples, return a point in latent space. """
    ...

def transform(examples):
    """ Take example, return reconstruction by decoder. """
    ...

x_test = ... # some examples to plot
x = ... # To plot for the x-axis
y_test = ... # Some ground truth properties, for colour coding
def outliers = ... # some extra, interesting examples (optional)
latent_dims = 3 # Or whatever

model Model(encode, transform, latent_dims, x, x_test, y_test, outliers)
model.run()
```
