import h5py

class h5Handler(object):

    def __init__(self, h5_path):
        self.h5_path = h5_path
        self.fid = h5py.File(h5_path, 'r')

    def read(self, key, start, end, step):
        return self.fid[key][start:end:step]

    