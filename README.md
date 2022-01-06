# Growing Neural Gas (GNG)
Based on this [Paper][id]. As input for the fit method a two dimensional numpy array is required. The dimension can be arbitrary. Upon completation, the neurons coordinates, the activity flag (whether the neuron is used or not) and the adjacency matrix is return. Only dependencies are Numpy and Matplotlib ( for plotting)


![Neural Gas Gif](./gng.gif)

## Example:

    import numpy
    from pygng.gng import PyGng

    data = np.random.normal(size=(500,2))
    gng = PyGNG(maxNeurons=100, ageMax=25, iterMax=25000)
    gng.fit(data)




[id]: https://proceedings.neurips.cc/paper/1994/file/d56b9fc4b0f1be8871f5e1c40c0067e7-Paper.pdf  "Original Paper"

Author: cmb87