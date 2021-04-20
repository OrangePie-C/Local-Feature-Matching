import numpy as np
import sys

def match_features(features1, features2, x1, y1, x2, y2):
    """
    This function does not need to be symmetric (e.g. it can produce
    different numbers of matches depending on the order of the arguments).

    To start with, simply implement the "ratio test", equation 4.18 in
    section 4.1.3 of Szeliski. There are a lot of repetitive features in
    these images, and all of their descriptors will look similar. The
    ratio test helps us resolve this issue (also see Figure 11 of David
    Lowe's IJCV paper).

    For extra credit you can implement various forms of spatial/geometric
    verification of matches, e.g. using the x and y locations of the features.

    Args:
    -   features1: A numpy array of shape (n,feat_dim) representing one set of
            features, where feat_dim denotes the feature dimensionality
    -   features2: A numpy array of shape (m,feat_dim) representing a second set
            features (m not necessarily equal to n)
    -   x1: A numpy array of shape (n,) containing the x-locations of features1
    -   y1: A numpy array of shape (n,) containing the y-locations of features1
    -   x2: A numpy array of shape (m,) containing the x-locations of features2
    -   y2: A numpy array of shape (m,) containing the y-locations of features2

    Returns:
    -   matches: A numpy array of shape (k,2), where k is the number of matches.
            The first column is an index in features1, and the second column is
            an index in features2
    -   confidences: A numpy array of shape (k,) with the real valued confidence for
            every match

    'matches' and 'confidences' can be empty e.g. (0x2) and (0x1)
    """
    #############################################################################
    # TODO: YOUR CODE HERE

    #hyper parameter
    threshold = 0.9
    Sorting = True

    ratio = np.zeros([features1.shape[0], 1])
    indexer = np.zeros ([features1.shape[0], 1])
    indexer[:] = np.NaN
    #for order in range(0, 1):


    for order in range (0,features1.shape[0]):
        sys.stdout.write('\r' +"   "+ str(int(order/features1.shape[0]*100+1)) + "%")

        Loss = np.zeros([features2.shape[0], 1])
        for order2 in range (0,features2.shape[0]):
            Loss[order2, 0] = sum(abs(features1[order,:]-features2[order2,:]))
            '''
            for order3 in range(0,128):
                Loss[order2,0] = Loss[order2,0] + abs(features1[order,order3]-features2[order2,order3])
                #Loss[order2, 0] = Loss[order2, 0] + features1[order, order3] - features2[order2, order3]
            '''

        Loss2 = np.sort(Loss.flatten())
        #Loss2 = np.sort(Loss)

        a= Loss2[0]/Loss2[1]

        #print(a)
        if a <threshold:
            ratio[order, 0] = a
            indexer[order, 0] = np.argmin(Loss)
    print()

    dummy = np.arange(indexer.shape[0]).T
    dummy = np.reshape(dummy,(dummy.shape[0],1))

    indexer = np.concatenate((dummy,indexer,ratio),axis=1)

    for num in range(indexer.shape[0]-1, -1,-1):
        if indexer [num,2]==0:
            indexer = np.delete(indexer,num,0)

    if Sorting ==True:
        indexer = indexer[indexer[:,2].argsort()]

    matches = indexer[:,0:2].astype(int)
    confidences = indexer[:,2]


    #print(matches)










        #############################################################################

    #raise NotImplementedError('`match_features` function in ' +
    #    '`student_feature_matching.py` needs to be implemented')

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return matches, confidences

