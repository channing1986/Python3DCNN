import tensorflow as tf

class DataInput_txt(object):
    def read_labeled_image_list(self,image_list_file):
    #"""Reads a .txt file containing pathes and labeles
    #Args:
    #   image_list_file: a .txt file with one /path/to/image per line
    #   label: optionally, if set label will be pasted after each line
    #Returns:
    #   List with all filenames in file image_list_file
    #"""
        f = open(image_list_file, 'r')
        filenames = []
        labels = []
        for line in f:
            filename, label = line[:-1].split(' ')
            filenames.append(filename)
            labels.append(int(label))
        return filenames, labels

    def read_images_from_disk(self,input_queue):
    #"""Consumes a single filename and label as a ' '-delimited string.
    #Args:
    #  filename_and_label_tensor: A scalar string tensor.
    #Returns:
    #  Two tensors: the decoded image, and the string label.
    #"""
        label = input_queue[1]
        file_contents = tf.read_file(input_queue[0])
        example = tf.image.decode_image(file_contents, channels=3)
        return example, label

    def load_image_lab_from_txt(self,filename):
# Reads pfathes of images together with their labels
        image_list, label_list = self.read_labeled_image_list(filename)

        images = tf.convert_to_tensor(image_list, dtype=tf.string)
        labels = tf.convert_to_tensor(label_list, dtype=tf.int32)

        # Makes an input queue
        input_queue = tf.train.slice_input_producer([images, labels],
                                                    num_epochs=1,
                                                    shuffle=True)

        image, label = self.read_images_from_disk(input_queue)
        return image,label
## Optional Preprocessing or Data Augmentation
## tf.image implements most of the standard image augmentation
#image = preprocess_image(image)
#label = preprocess_label(label)

## Optional Image and Label Batching
#image_batch, label_batch = tf.train.batch([image, label],
#                                          batch_size=batch_size)


