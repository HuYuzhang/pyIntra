import h5py

class h5Handler(object):

    def __init__(self, h5_path):
        self.h5_path = h5_path

    def read(self, key, start, end, step):
        fid = h5py.File(self.h5_path, 'r')
        ret = fid[key][start:end:step]
        fid.close()
        return ret

    # right now very bad way to assign 3072 and 1024, but not a big problem
    # assume that datas and labels are of size [n, c, h, w]
    def write(self, datas, labels, create=True):
        if create:
            f = h5py.File(self.h5_path, 'w')
            f.create_dataset('data', data=datas, maxshape=(None, 3072, 1, 1), chunks=True, dtype='float32')
            f.create_dataset('label', data=labels, maxshape=(None, 1024, 1, 1), chunks=True, dtype='float32')
            f.close()
        else:
            # append mode
            f = h5py.File(self.h5_path, 'a')
            h5data = f['data']
            h5label = f['label']
            cursize = h5data.shape
            addsize = datas.shape

            # # --------------for debug------------------
            # print('-------now begin to add data------')
            # print(cursize)
            # # --------------for debug------------------

            h5data.resize([cursize[0] + addsize[0], 3072, 1, 1])
            h5label.resize([cursize[0] + addsize[0], 1024, 1, 1])
            h5data[-addsize[0]:,:,:,:] = datas
            h5label[-addsize[0]:,:,:,:] = labels
            f.close()