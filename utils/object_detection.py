import mxnet as mx
import gluoncv as gcv
import matplotlib.pyplot as plt


def detect_objects(network: gcv.model_zoo.HybridBlock, image):
    """Displays the object classification image predicted by the given network.

        Parameters
        ----------
        network : GluonCV HybridBlock trained to recognize objects
        image : The image array in [Batch Number, Color, Height, Width] format
    """
    image, chw_image = gcv.data.transforms.presets.yolo.transform_test(image, short=512)
    prediction = [array[0] for array in network(image)]
    class_indices, probabilities, bounding_boxes = prediction
    gcv.utils.viz.plot_bbox(chw_image,
                            bounding_boxes,
                            probabilities,
                            class_indices,
                            class_names=network.classes)
    plt.show()

if __name__ == "__main__":
    image = mx.image.imread('../images/dog.jpg')
    network = gcv.model_zoo.get_model('yolo3_darknet53_coco', pretrained=True)
    detect_objects(network, image)
