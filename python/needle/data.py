import gzip
import os
import pickle
import struct
from typing import Any, Iterable, Iterator, List, Optional, Sized, Union

import numpy as np
from needle import backend_ndarray as nd

from .autograd import Tensor


def unpack_part(fmt, data):
    size = struct.calcsize(fmt)
    return struct.unpack(fmt, data[:size]), data[size:]

def read_idx_file(filename):
    with gzip.open(filename, mode='rb') as fileobj:
        data = fileobj.read()

        (zero1, zero2, type_id, dims), data = unpack_part('>bbbb', data)
        if zero1 != 0 or zero2 != 0:
            raise Exception("Invalid file format")

        types = {
            int('0x08', base=16): 'B',
            int('0x09', base=16): 'b',
            int('0x0B', base=16): 'h',
            int('0x0C', base=16): 'i',
            int('0x0D', base=16): 'f',
            int('0x0E', base=16): 'd'
        }
        type_code = types[type_id]

        dim_sizes, data = unpack_part('>' + ('i' * dims), data)
        num_examples = dim_sizes[0]
        input_dim = int(np.prod(dim_sizes[1:]))

        X, data = unpack_part('>' + (type_code * (num_examples * input_dim)), data)
        if data:
            raise Exception("invalid file format")

        new_shape = (num_examples, input_dim) if input_dim > 1 else num_examples
        return np.array(X).reshape(new_shape, order='C')

def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    images = read_idx_file(image_filename).astype('float32')
    images = (images - images.min()) / (images.max() - images.min())
    labels = read_idx_file(label_filename).astype('uint8')
    return images, labels
    ### END YOUR SOLUTION

class Transform:
    def __call__(self, x):
        raise NotImplementedError

class Normalize(Transform):
    def __init__(self, base):
        self.base = base
    
    def __call__(self, img):
        result = img / self.base
        return result

class RandomFlipHorizontal(Transform):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        if flip_img:
            result = np.flip(img, axis=(1,))
        else:
            result = img
        return result
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return
            H x W x C NDArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(
            low=-self.padding, high=self.padding + 1, size=2
        )
        ### BEGIN YOUR SOLUTION
        padding = ((self.padding, self.padding), (self.padding, self.padding), (0,0))
        img_padded = np.pad(img, padding, 'constant', constant_values=0)
        x_from = self.padding + shift_x
        x_to = x_from + img.shape[0]
        y_from = self.padding + shift_y
        y_to = y_from + img.shape[1]
        result = img_padded[x_from:x_to, y_from:y_to, :]
        return result
        ### END YOUR SOLUTION


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
    """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(
                np.arange(len(dataset)), range(batch_size, len(dataset), batch_size)
            )
        # else:
        #     # print("DEBUG[__init__]: Reshuffling data...")
        #     ordering = np.arange(len(self.dataset))
        #     np.random.shuffle(ordering)
        #     batch_ranges = range(self.batch_size, len(self.dataset), self.batch_size)
        #     self.ordering = np.array_split(ordering, batch_ranges)

    def __iter__(self):
        ### BEGIN YOUR SOLUTION
        self.batch_idx = 0
        if self.shuffle:
            # print("DEBUG[__iter__]: Reshuffling data...")
            ordering = np.arange(len(self.dataset))
            np.random.shuffle(ordering)
            batch_ranges = range(self.batch_size, len(self.dataset), self.batch_size)
            self.ordering = np.array_split(ordering, batch_ranges)
        ### END YOUR SOLUTION
        return self

    def __next__(self):
        ### BEGIN YOUR SOLUTION
        if self.batch_idx < len(self.ordering):
            idx = self.ordering[self.batch_idx]
            self.batch_idx += 1
            result = self.dataset[idx]
            result = tuple([Tensor(x) for x in result])
            return result
        else:
            raise StopIteration
        ### END YOUR SOLUTION


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms)
        self.X, self.Y = parse_mnist(image_filename, label_filename)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        X_items = self.X[index]
        Y_items = self.Y[index]
        if isinstance(index, (slice, np.ndarray)):
            Y_items = np.reshape(Y_items, (Y_items.shape[0]))
            X_items = np.reshape(X_items, (X_items.shape[0], 28, 28, 1))
            for item_idx in range(X_items.shape[0]):
                X_item = X_items[item_idx]
                X_items[item_idx] = self.apply_transforms(X_item)
        else:
            # Y_items = Y_items[0]
            X_items = np.reshape(X_items, (28, 28, 1))
            X_items = self.apply_transforms(X_items)
        return (X_items, Y_items)
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.X.shape[0]
        ### END YOUR SOLUTION


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms)
        
        batches = []
        if train:
            for batch_id in range(1, 6):
                batch = unpickle(os.path.join(base_folder, f"data_batch_{batch_id}"))
                batches.append(batch)
        else:
            batch = unpickle(os.path.join(base_folder, "test_batch"))
            batches.append(batch)
        
        self.X = np.concatenate([batch[b'data'] for batch in batches])
        self.X = np.reshape(self.X, (self.X.shape[0], 3, 32, 32)) / 255
        self.y = np.concatenate([batch[b'labels'] for batch in batches])

        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        X_items = self.X[index]
        Y_items = self.y[index]
        if isinstance(index, (slice, np.ndarray)):
            Y_items = np.reshape(Y_items, (Y_items.shape[0]))
            X_items = np.reshape(X_items, (X_items.shape[0], 3, 32, 32))
            for item_idx in range(X_items.shape[0]):
                X_item = X_items[item_idx]
                X_items[item_idx] = self.apply_transforms(X_item)
        else:
            X_items = np.reshape(X_items, (3, 32, 32))
            X_items = self.apply_transforms(X_items)
        return (X_items, Y_items)
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return self.X.shape[0]
        ### END YOUR SOLUTION


class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])






class Dictionary(object):
    """
    Creates a dictionary from a list of words, mapping each word to a
    unique integer.
    Attributes:
    word2idx: dictionary mapping from a word to its unique ID
    idx2word: list of words in the dictionary, in the order they were added
        to the dictionary (i.e. each word only appears once in this list)
    """
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        """
        Input: word of type str
        If the word is not in the dictionary, adds the word to the dictionary
        and appends to the list of words.
        Returns the word's unique ID.
        """
        ### BEGIN YOUR SOLUTION
        word_idx = self.word2idx.get(word, None)
        if word_idx is None:
            word_idx = len(self.idx2word)
            self.idx2word.append(word)
            self.word2idx[word] = word_idx
        return word_idx
        ### END YOUR SOLUTION

    def __len__(self):
        """
        Returns the number of unique words in the dictionary.
        """
        ### BEGIN YOUR SOLUTION
        return len(self.idx2word)
        ### END YOUR SOLUTION



class Corpus(object):
    """
    Creates corpus from train, and test txt files.
    """
    def __init__(self, base_dir, max_lines=None):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(base_dir, 'train.txt'), max_lines)
        self.test = self.tokenize(os.path.join(base_dir, 'test.txt'), max_lines)

    def tokenize(self, path, max_lines=None):
        """
        Input:
        path - path to text file
        max_lines - maximum number of lines to read in
        Tokenizes a text file, first adding each word in the file to the dictionary,
        and then tokenizing the text file to a list of IDs. When adding words to the
        dictionary (and tokenizing the file content) '<eos>' should be appended to
        the end of each line in order to properly account for the end of the sentence.
        Output:
        ids: List of ids
        """
        ### BEGIN YOUR SOLUTION
        eos_id = self.dictionary.add_word('<eos>')
        cur_line = 0
        result = []
        with open(path, 'rb') as file:
            for line in file.readlines():
                # print(f"line[{cur_line}]: {line=}")
                for word in line.split():
                    word_id = self.dictionary.add_word(word)
                    # print(f"\tgetting word id for {word} -> {word_id=}")
                    result.append(word_id)
                result.append(eos_id)
                cur_line += 1
                if max_lines and cur_line >= max_lines:
                    break
        return result
        ### END YOUR SOLUTION


def batchify(data, batch_size, device, dtype):
    """
    Starting from sequential data, batchify arranges the dataset into columns.
    For instance, with the alphabet as the sequence and batch size 4, we'd get
    ┌ a g m s ┐
    │ b h n t │
    │ c i o u │
    │ d j p v │
    │ e k q w │
    └ f l r x ┘.
    These columns are treated as independent by the model, which means that the
    dependence of e. g. 'g' on 'f' cannot be learned, but allows more efficient
    batch processing.
    If the data cannot be evenly divided by the batch size, trim off the remainder.
    Returns the data as a numpy array of shape (nbatch, batch_size).
    """
    ### BEGIN YOUR SOLUTION
    nbatch = len(data) // batch_size
    last_idx = nbatch * batch_size
    data_subset = data[:last_idx]
    data_subset_np = np.array(data_subset, dtype=dtype)
    data_subset_np = data_subset_np.reshape((batch_size, nbatch)).T
    return data_subset_np
    ### END YOUR SOLUTION


def get_batch(batches, i, bptt, device=None, dtype=None):
    """
    get_batch subdivides the source data into chunks of length bptt.
    If source is equal to the example output of the batchify function, with
    a bptt-limit of 2, we'd get the following two Variables for i = 0:
    ┌ a g m s ┐ ┌ b h n t ┐
    └ b h n t ┘ └ c i o u ┘
    Note that despite the name of the function, the subdivison of data is not
    done along the batch dimension (i.e. dimension 1), since that was handled
    by the batchify function. The chunks are along dimension 0, corresponding
    to the seq_len dimension in the LSTM or RNN.
    Inputs:
    batches - numpy array returned from batchify function
    i - index
    bptt - Sequence length
    Returns:
    data - Tensor of shape (bptt, bs) with cached data as NDArray
    target - Tensor of shape (bptt*bs,) with cached data as NDArray
    """
    ### BEGIN YOUR SOLUTION

    rem_len = len(batches) - i
    rem_bptt = min(bptt+1, rem_len)-1
    assert(rem_bptt >= 2)

    data_i = i
    data_j = i + rem_bptt
    target_i = i + 1
    target_j = i + rem_bptt + 1

    data_np = batches[data_i:data_j, :]
    target_np = batches[target_i:target_j, :]
    data_t = Tensor(data_np, device=device, dtype=dtype, requires_grad=False)
    target_t = Tensor(target_np.flatten(), device=device, dtype=dtype, requires_grad=False)
    return data_t, target_t
    ### END YOUR SOLUTION
