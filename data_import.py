import struct
import os
import progressbar
import numpy as np


class dataset:
    def __init__(self, imagesfile, labelsfile):
        self.__data_import(imagesfile, labelsfile)

    @staticmethod
    def __get_one_hot(value, deepness):
        assert value <= deepness, "Loolz" 
        
        a = np.zeros([deepness])
        a[value] = 1.0
        return a
    
    def __data_import(self, filename, labelfile):
        with open(filename, 'rb') as file:
            with open(labelfile, 'rb') as lfile:
                magik, size, rows, cols = struct.unpack(">IIII", file.read(16))
                lmagik, lsize = struct.unpack(">II", lfile.read(8))

                assert lsize == size, "Loul" 

                image = []
                label = []

                bar = progressbar.ProgressBar()
                for i in bar(range(size), max_value=size):
                    image.append(struct.unpack(">"+"B"*(rows*cols), file.read(rows*cols)))
                    label.append(
                        dataset.__get_one_hot(
                            struct.unpack(">B", lfile.read(1))[0],
                            10
                        )   
                    )

                self.__images = np.float32(image).reshape([size, rows, cols, 1]) / 255.0
                self.__labels = np.float32(label)

    @property
    def labels_dtype(self):
        return self.__labels.dtype

    @property
    def images_dtype(self):
        return self.__images.dtype

    @property
    def size(self):
        return self.__labels.shape[0]

    @property
    def shape(self):
        return ((self.__images.shape[1], self.__images.shape[2], self.__images.shape[3]), self.__labels.shape[1])

    def get_random_batcher(self, batch_size):
        while True:
            index = np.arange(self.__labels.shape[0])
            np.random.shuffle(index)

            s_labels, s_images = self.__labels[index], self.__images[index] 
            for i in range(self.__labels.shape[0] // batch_size):
                yield (s_images[i*batch_size:(i+1)*batch_size], s_labels[i*batch_size:(i+1)*batch_size])

    def get_batcher(self, batch_size):
        if self.__labels.shape[0] % batch_size != 0:
            print("[/!\ Warning /!\] the full set will not be executed because of a poor choice of batch_size")
        
        for i in range(self.__labels.shape[0] // batch_size):
            yield self.__images[i*batch_size:(i+1)*batch_size], self.__labels[i*batch_size:(i+1)*batch_size]


class mnist_dataset: 
    def __init__(self, path):
        train_images_path = os.path.join(path, "train-images.idx3-ubyte")
        train_lables_path = os.path.join(path, "train-labels.idx1-ubyte")
        eval_images_path = os.path.join(path, "t10k-images.idx3-ubyte")
        eval_lables_path = os.path.join(path, "t10k-labels.idx1-ubyte")

        print("Importing Training set...")
        self.train = dataset(train_images_path, train_lables_path)

        print("Importing Evaluation set...")
        self.eval = dataset(eval_images_path, eval_lables_path)

    @property
    def labels_dtype(self):
        assert self.train.labels_dtype == self.eval.labels_dtype, "loul"       
        return self.train.labels_dtype

    @property
    def images_dtype(self):
        assert self.train.images_dtype == self.eval.images_dtype, "loul"       
        return self.train.images_dtype

    @property
    def shape(self):
        assert self.train.shape == self.eval.shape, "loul"       
        return self.train.shape


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    mnist = mnist_dataset("/tmp/tensorflow/mnist/input_data")

    label, image = next(mnist.train.get_batch(1))
    print(label)
    plt.imshow(image[0, :], cmap=plt.get_cmap("gray"))
    plt.show()
