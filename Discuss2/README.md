# Dependencies

What is new from Discuss1:

	1. In <models.py>, two classes "SigmoidNormNet, ReLUNormNet" are added. They are for experimenting the network performance with normalization layers. Before experimenting, you should find the code of these two classes and set "norm_layer" variable to the correct normalization type.
	2. Jupyter Notebook <main_grad_analysis.py.ipynb> is added. This is the main script to visualize the gradient flow during training.
	3. New models are imported in <main.py>, so that the regression result of models with normalization can also be evaluated.
