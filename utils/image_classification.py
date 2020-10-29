import gluoncv as gcv
import mxnet as mx
import numpy as np


def print_topk(network: gcv.model_zoo.HybridBlock, image, k: int) -> None:
    """Print the top k predicted classes for an image.

    Parameters
    ----------
    network : GluonCV HybridBlock trained to classify images
    image : The image array in [Batch Number, Channel, Height, Width] format
    k : the top k classifications
    """
    prediction = network(image)[0]
    mx.nd.topk(prediction, k=k)

    topk_indices = mx.nd.topk(prediction, k=k)
    for i in range(k):
        class_index = topk_indices[i].astype('int').asscalar()
        class_label = network.classes[class_index]
        class_probability = prediction[class_index]
        print('#{} {} ({:0.3}%)'.format(i + 1, class_label, class_probability.asscalar() * 100))
