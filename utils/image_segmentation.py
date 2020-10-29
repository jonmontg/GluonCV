from mxnet.gluon.data.vision import transforms
import mxnet as mx
import gluoncv as gcv
import matplotlib.pyplot as plt


# transform an image into a Tensor for image segmentation based on the
# statistics from the ImageNet 1k dataset.
transform_fn = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([.485, .456, .406], [.229, .224, .225])
])


# Semantic Segmentation
def show_image_segmentation(network: gcv.model_zoo.HybridBlock, image) -> None:
    image = transform_fn(image)
    image = image.expand_dims(0)

    # output = [Class, Width, Height]
    output = network.predict(image)[0]
    prediction = mx.nd.argmax(output, 0).asnumpy()
    prediction_image = gcv.utils.viz.get_color_pallete(prediction, 'ade20k')
    plt.imshow(prediction_image)
    plt.show()


def get_pixel_class_probabilities(network: gcv.model_zoo.HybridBlock, image) -> list:
    image = transform_fn(image)
    image = image.expand_dims(0)
    # output = [Class, Width, Height]
    output = network.predict(image)[0]
    # get the segmentation probabilities for all pixels across classes
    return mx.nd.softmax(output, axis=0)




# Instance Segmentation
