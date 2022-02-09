import numpy as np


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
    # TODO: YOUR CODE HERE                                                        #
    #############################################################################

    features1 = features1**.7
    features2 = features2**.7

    # Result matrix
    fv_distance = np.zeros((features1.shape[0], features2.shape[0]))
    feature1_index = []
    feature2_index = []
    feature1_2_distance = []

    # Step 1.Iterate through features1(k1)
    for k1 in range(features1.shape[0]):

        # Step 2.Iterate through features2(k2)
        for k2 in range(features2.shape[0]):
            feature1 = features1[[k1],:]
            feature2 = features2[[k2],:]

            # Step 3.Compute Euclidean distance between feature1(k1) and feature2(k2) 
            feature_distance = np.sqrt(np.sum((feature1 - feature2)**2))
            fv_distance[k1,k2] = feature_distance

        # Step 4.Obtain the indexes of first and second minimum distances
        feature2_sorted_index = np.argsort(fv_distance[k1,:])
        min_ind_1 = feature2_sorted_index[0]
        min_ind_2 = feature2_sorted_index[1]
        feat_dis_min1 = fv_distance[k1,min_ind_1]
        feat_dis_min2 = fv_distance[k1,min_ind_2]

        # Step 5.Compute the ratio between first and second minimum distances 
        ratio = feat_dis_min1/feat_dis_min2


        # Step 6.Accept feature1(k1) and feature2(min_ind_1) as matching features if ratio < 0.8
        if ratio < 0.9:  #Change to 0.9 while running Mount Rushmore
            feature1_index.append(k1)
            feature2_index.append(min_ind_1)
            feature1_2_distance.append(feat_dis_min1)


    # Step 7.Arrange and save matches and confidences
    matches = np.stack((np.asarray(feature1_index), np.asarray(feature2_index)), axis = -1)
    confidences = np.asarray(feature1_2_distance)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return matches, confidences
