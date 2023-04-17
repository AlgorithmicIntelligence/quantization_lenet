from __future__ import ( division, absolute_import, print_function, unicode_literals )

import gzip
import os
import sys
import numpy as np

if sys.version_info >= (3,):
    import urllib.request as urllib2
    import urllib.parse as urlparse
else:
    import urllib2
    import urlparse

def download(url, dest="./MNIST/"):
    if not os.path.isdir(dest):
        os.mkdir(dest)
    """ 
    Download and save a file specified by url to dest directory,
    """
    u = urllib2.urlopen(url)

    scheme, netloc, path, query, fragment = urlparse.urlsplit(url)
    filename = os.path.basename(path)
    if not filename:
        filename = 'downloaded.file'
    file_path = os.path.join(dest,filename)
    if os.path.isfile(file_path):
        extract_gz(file_path)
    else:
        with open(file_path, 'wb') as f:
            meta = u.info()
            meta_func = meta.getheaders if hasattr(meta, 'getheaders') else meta.get_all
            meta_length = meta_func("Content-Length")
            file_size = None
            if meta_length:
                file_size = int(meta_length[0])
            print("Downloading: {0} Bytes: {1}".format(url, file_size))

            file_size_dl = 0
            block_sz = 8192
            while True:
                buffer = u.read(block_sz)
                if not buffer:
                    break

                file_size_dl += len(buffer)
                f.write(buffer)

                status = "{0:16}".format(file_size_dl)
                if file_size:
                    status += "   [{0:6.2f}%]".format(file_size_dl * 100 / file_size)
                status += chr(13)
                print(status, end="")
            print()
        extract_gz(file_path)


def extract_gz(filename):
    with gzip.open(filename, 'rb') as infile:
        with open(filename[0:-3], 'wb') as outfile:
            for line in infile:
                outfile.write(line)


if __name__ == "__main__":
    url1 = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
    url2 = "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
    url3 = "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
    url4 = "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"

    if not os.path.isdir("MNIST"):
        os.mkdir("MNIST")
    os.chdir("./MNIST")

    print("Downloading MNIST dataset...")
    f1 = download(url1)
    f2 = download(url2)
    f3 = download(url3)
    f4 = download(url4)

    print("\nExtracting...")
    extract_gz("./"+f1)
    extract_gz("./"+f2)
    extract_gz("./"+f3)
    extract_gz("./"+f4)

    print("\nDeleting .gz files...")
    os.remove(f1)
    os.remove(f2)
    os.remove(f3)
    os.remove(f4)
    print("\nMNIST dataset is stored in the 'MNIST' folder")


def load():
    int_type = np.dtype('int32').newbyteorder('>')
    nMetaDataBytes = 4 * int_type.itemsize

    if not os.path.isfile("./MNIST/train-images-idx3-ubyte"):
        download("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")
    data = np.fromfile('./MNIST/train-images-idx3-ubyte', dtype='ubyte')
    magicBytes, nImages, width, height = np.frombuffer(data[:nMetaDataBytes].tobytes(), int_type)

    train_data = data[nMetaDataBytes:].astype(dtype='float32').reshape([nImages, width, height])
    if not os.path.isfile("./MNIST/train-labels-idx1-ubyte"):
        download("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")
    train_labels = np.fromfile('./MNIST/train-labels-idx1-ubyte', dtype='ubyte')[2 * int_type.itemsize:]

    if not os.path.isfile("./MNIST/t10k-images-idx3-ubyte"):
        download("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")
    data = np.fromfile('./MNIST/t10k-images-idx3-ubyte', dtype='ubyte')
    magicBytes, nImages, width, height = np.frombuffer(data[:nMetaDataBytes].tobytes(), int_type)

    test_data = data[nMetaDataBytes:].astype(dtype='float32').reshape([nImages, width, height])
    if not os.path.isfile("./MNIST/t10k-labels-idx1-ubyte"):
        download("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")
    test_labels = np.fromfile('./MNIST/t10k-labels-idx1-ubyte', dtype='ubyte')[2 * int_type.itemsize:]

    return train_data, train_labels, test_data, test_labels

