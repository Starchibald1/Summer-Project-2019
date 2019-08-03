import numpy as np

def process_data(data_list):
    '''
     :param data_list: List of data, input as one long string
     Function converts a string into a list of required information. It does
     this in several steps:
     First bit is conversion of long string in 'data' into three lists of data:
        1) Create empty lists for storage of data
        2) Loop through the for loop, using the .split function to split at each
           requested character, with the function output being a list
        3) Convert each list into a numpy array (will be useful later) and 
           converts required data from string to float/int
     Returns a 3-tuple of lists - id_list, attribute_list, truth_list, with
     the following dimensions:
        1) id_list --> (N,)
        2) attribute_list --> (N, 30)
        3) truth_list --> (N,)
    '''
    # Use empty arrays so that they can be appended to easily
    id_list = []
    attribute_list = []
    truth_list = []
    
    for data_ in data_list:
        all_ = data_.split(',')
        id_ = all_[0]
        truth = all_[1]
        attribute = all_[2:]
        
        id_list.append(id_)
        truth_list.append(truth)
        attribute_list.append(attribute)
    
    id_list = np.array(id_list).astype(np.int)
    attribute_list = np.array(attribute_list).astype(np.float32)
    truth_list = np.array(truth_list)
    
    return id_list, attribute_list, truth_list


def softmax(z):
    z_exp = np.exp(z)
    z_sum = np.sum(np.exp(z))
    return z_exp / z_sum

    
def crossEntropy(y_pred, y):
    """
    y_pred = predicted values
    y = truth values
    """
    
    size_y = np.shape(y)[0]
    y_pred = np.reshape(y_pred, (1, size_y))
    y = np.reshape(y, (size_y, 1))
    return -(1/size_y)*(np.dot(np.log(y_pred), y) + np.dot((1 - y), np.log(1 - y_pred)))


def gradError():
    
    return


def prediction(data, labels):
    W = np.random.random(size=(np.shape(attribute_list)[1], 1))
    b = np.random.random()
    
    y = softmax(np.dot(data, W) + b)
    error = crossEntropy(y, labels)    
    
    
    
    return

# Opens the file with the data and stores all the data as a single long string
# in 'data' variable
with open('wdbc_txt.txt', 'r') as file:
    data = file.read()
data_list = data.split('\n')[:-1]

# Output is a 3-tuple - it can be written either as (a, b, c) or a, b, c as the
# output
id_list, attribute_list, truth_list = process_data(data_list)

np.random.seed(100)

# Printing the output in three different ways, using np.shape to get the 
# dimensions of each list structure
print('ID_list dimensions: {0}'.format(np.shape(id_list)))
print('Attribute List: ', np.shape(attribute_list))
print('Truth List: ' + str(np.shape(truth_list)))

prediction()



