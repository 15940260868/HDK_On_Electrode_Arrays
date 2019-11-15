import numpy as np

def loadtxtAndcsv_data(fileName, split, dataType):
    return np.loadtxt(fileName, delimiter=split, dtype=dataType)

def savetxt(filename,x):
    np.savetxt(filename,x,fmt="%.2f", newline='\n')

 # if length >= 0:
 #            length = int(length)

def random_list(min, max, length):
    min, max = (float(min), float(max)) if min <= max else (float(max), float(min))

    random_list = []
    for i in range(length):
        random_list.append(min+np.round(np.random.rand(),2) * (max-min))

    random_array = np.array(random_list)
    # print("random_list: {}".format(len(random_list)))
    random_array = np.reshape(random_array, (64, 64))
    return random_array


if __name__ == '__main__':
    #data = loadtxtAndcsv_data("./input_64.txt", "\t", np.float64)
    #processed = np.reshape(np.array(data), (64, 64))
    processed = np.ones((64, 64))
    print("original array:\n {}" .format(processed))
    # print("shape: {} ".format(processed.shape))

    for i in range(3):
        a = random_list(0.01, 0.05, 4096)
        randomArray = processed * a
        print("\nAfter mutiply 1-64 random num:\n {}" .format(randomArray))
        #np.savetxt("./newInput_%d.txt" % (i+317), randomArray, fmt="%.6f", delimiter=" ")

