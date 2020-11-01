import mxnet as mx
import gluoncv as gcv
import matplotlib.pyplot as plt


def show_object_detection_image(network: gcv.model_zoo.HybridBlock, image: mx.nd.NDArray):
    """Displays the object classification image predicted by the given network.

        Parameters
        ----------
        network : GluonCV HybridBlock trained to recognize objects
        image : The image array in [Batch Number, Color, Height, Width] format
    """
    image, chw_image = gcv.data.transforms.presets.yolo.transform_test(image, short=512)
    class_indices, probabilities, bounding_boxes = detect_objects(network, image)
    gcv.utils.viz.plot_bbox(chw_image,
                            bounding_boxes,
                            probabilities,
                            class_indices,
                            class_names=network.classes)
    plt.show()


def detect_objects(network: gcv.model_zoo.HybridBlock, one_image_batch: mx.nd.NDArray):
    """Returns the class indices, predicted probabilities, and the bounding boxes of the objects detected in the image.

    :param network: GluonCV HybridBlock trained to recognize objects
    :param one_image_batch: The image array in [Batch Number, Color, Height, Width] format
    :return: (class indices, probabilities, bounding boxes)
    """
    prediction = [array[0] for array in network(one_image_batch)]
    class_indices, probabilities, bounding_boxes = prediction
    return class_indices, probabilities, bounding_boxes


def count_object(network: gcv.model_zoo.HybridBlock, one_image_batch: mx.nd.NDArray, detection_threshold: float, object_label: str):
    """
    Counts the number of instances of the given object appear in the image.
    :param network: GluonCV HybridBlock trained to recognize objects
    :param one_image_batch: The image array in [Batch Number, Color, Height, Width] format
    :param object_label: The label of the object to count
    :return: The number of predicted instances of the object
    """
    class_ids, probabilities, _ = detect_objects(network, one_image_batch)
    # get the class index of the object we are looking for
    label_id = network.classes.index(object_label)

    # find the predictions that meet the detection threshold and the object labels that we are looking for
    meets_threshold = mx.nd.ndarray.greater_equal(probabilities, detection_threshold).flatten()[0]
    label_predictions = mx.nd.ndarray.equal(class_ids, label_id).flatten()[0]

    return mx.nd.dot(meets_threshold, label_predictions).asscalar()
