"""
Testing out the format of datasets.
run with -i option to keep interpreter open
ex: python -i data_tester.py (args)
"""
import time
from options.train_options import TrainOptions
from data import create_dataset

print("Use the following command to get the data")
print("dataset, img_dict = get_data()")
print("or")
print("dataset, img_dict = data_tester.get_data()")
print("depending on how you imported")

def get_data():
    """
    dataset, img_dict = get_data()
    """
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    img_dict = dataset.__iter__().__next__()
    print("'dataset.__len__()': %d" % dataset.__len__())
    print("img_dict is a dictionary generated with 'img_dict = dataset.__iter__().__next__()'")
    print("Use it with keys %s" % str(img_dict.keys()))
    print("Returning tuple of (dataset, img_dict) for your use")
    return dataset, img_dict

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    img_dict = dataset.__iter__().__next__()
    print("'dataset.__len__()': %d" % dataset.__len__())
    print("img_dict is a dictionary generated with 'img_dict = dataset.__iter__().__next__()'")
    print("Use it with keys %s" % str(img_dict.keys()))