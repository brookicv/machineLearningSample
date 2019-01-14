# import the necessary packages
import h5py
import os


class HDF5DatasetWriter:
    def __init__(self, dims, outputPath, dataKey='images', bufSize=1000):
        '''
        check to see if the output path exists, and if so, raise an exception
        '''
        if os.path.exists(outputPath):
            raise ValueError("The supplied 'output path' already exists and cannot be overwritten. Manually delete the file before contuinuing", outputPath)

        # open the HDF5 database for writing and create two datasets: one to store the images/feature and another to store class labels
        self.db = h5py.File(outputPath, 'w')
        self.labels = self.db.create_dataset('labels', (dims[0],), dtype="int")
        self.data = self.db.create_dataset(dataKey,dims,dtype="float")

        # store the buffer size, then initizlize the buffer itself along with the index into the database
        self.bufSize = bufSize
        self.buffer = {'data': [], 'labels': []}
        self.idx = 0

    def add(self, rows, labels):
        '''
        add the rows and labels to the buffer
        '''
        self.buffer['data'].extend(rows)
        self.buffer['labels'].extend(labels)

        # check to see if the buffer needs to be flused to disk
        if len(self.buffer['data']) >= self.bufSize:
            self.flush()

    def flush(self):
        '''
        write the buffers to disk then reset the buffer
        '''
        i = self.idx + len(self.buffer['data'])
        self.data[self.idx:i] = self.buffer['data']
        self.labels[self.idx:i] = self.buffer['labels']
        self.idx = i
        self.buffer = {'data': [], 'labels': []}

    def storeClassLabels(self, classLabels):
        '''
        create a dataset to store the actual class label names, then store the class labels
        '''
        dt = h5py.special_dtype(vlen=str)
        labelSet = self.db.create_dataset('label_names', (len(classLabels),), dtype=dt)
        labelSet[:] = classLabels

    def close(self):
        '''
        check to see if ther is any other entries in the buffer that need to be flushed to the disk
        '''
        if len(self.buffer['data']) > 0:
            self.flush()

        # close the dataset
        self.db.close()
