import numpy as np

def preprocess_data(all_data):
    """

    :param all_data: dataframe of data read in from "student-por.csv"
    :return: a 2D numpy array where all_data now consists of integers

    """
    np_data = all_data.iloc[:,0].values.reshape(-1, 1)
    np_data = np.array([row[0].split(",") for row in np_data])

    X = np_data[:,1:32]
    X[:,0] = (X[:,0] == "\"F\"").astype(int)
    # np.array([1 if x[1:-1] == "F" else 0 for x in X[:,0]])
    # X[:,2] = np.array([1 if x[1:-1] == "R" else 0 for x in X[:,2]])
    X[:,2] = (X[:,2] == "\"R\"").astype(int)
    # X[:,3] = np.array([1 if x[1:-1] == "GT3" else 0 for x in X[:,3]])
    X[:,3] = (X[:,3] == "\"GT3\"").astype(int)
    # X[:,4] = np.array([1 if x[1:-1] == "T" else 0 for x in X[:,4]])
    X[:,4] = (X[:,4] == "\"T\"").astype(int)
    X[:,7] = np.array([4 if x[1:-1] == "teacher" else
                       3 if x[1:-1] == "health" else
                       2 if x[1:-1] == "services" else
                       1 if x[1:-1] == "at_home" else 0
                       for x in X[:,7]])
    X[:,8] = np.array([4 if x[1:-1] == "teacher" else
                       3 if x[1:-1] == "health" else
                       2 if x[1:-1] == "services" else
                       1 if x[1:-1] == "at_home" else 0
                       for x in X[:,8]])
    X[:,9] = np.array([3 if x[1:-1] == "home" else
                        2 if x[1:-1] == "reputation" else
                        1 if x[1:-1] == "course" else 0
                        for x in X[:,9]])
    X[:,10] = np.array([2 if x[1:-1] == "mother" else
                        1 if x[1:-1] == "father" else 0
                        for x in X[:,10]])

    X[:,14:22] = (X[:,14:22] == "\"yes\"").astype(int)
    X[:,29] = np.array([x[1:-1] for x in X[:,29]])
    X[:,30] = np.array([x[1:-1] for x in X[:,30]])

    X = X.astype(int)
    y = np_data[:,32].astype(int)
    return X, y
