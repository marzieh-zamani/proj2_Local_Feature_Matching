import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_interest_points(image, feature_width):
    """
    Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
    You can create additional interest point detector functions (e.g. MSER)
    for extra credit.

    If you're finding spurious interest point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.

    Useful in this function in order to (a) suppress boundary interest
    points (where a feature wouldn't fit entirely in the image, anyway)
    or (b) scale the image filters being used. Or you can ignore it.

    By default you do not need to make scale and orientation invariant
    local features.

    The lecture slides and textbook are a bit vague on how to do the
    non-maximum suppression once you've thresholded the cornerness score.
    You are free to experiment. For example, you could compute connected
    components and take the maximum value within each component.
    Alternatively, you could run a max() operator on each sliding window. You
    could use this to ensure that every interest point is at a local maximum
    of cornerness.

    Args:
    -   image: A numpy array of shape (m,n,c),
                image may be grayscale of color (your choice)
    -   feature_width: integer representing the local feature width in pixels.

    Returns:
    -   x: A numpy array of shape (N,) containing x-coordinates of interest points
    -   y: A numpy array of shape (N,) containing y-coordinates of interest points
    -   confidences (optional): numpy nd-array of dim (N,) containing the strength
            of each interest point
    -   scales (optional): A numpy array of shape (N,) containing the scale at each
            interest point
    -   orientations (optional): A numpy array of shape (N,) containing the orientation
            at each interest point
    """
    confidences, scales, orientations = None, None, None
    #############################################################################
    # TODO: YOUR HARRIS CORNER DETECTOR CODE HERE                                                      #
    #############################################################################

    # Define alpha
    alpha = 0.04

    # Step 1: Compute the horizontal and vertical derivatives of the image (Ix and Iy) using Sobel filters  
    Ix = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=3)

    #Step 2: Compute  the outer products of gradients (Ixx, Iyy, Ixy)
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy

    #Step 3: Convolve each of these images with a larger Gaussian (ksize = 5) 
    Ixx = cv2.filter2D(Ixx, cv2.CV_64F, cv2.getGaussianKernel(ksize=5, sigma=1))
    Ixy = cv2.filter2D(Ixy, cv2.CV_64F, cv2.getGaussianKernel(ksize=5, sigma=1))
    Iyy = cv2.filter2D(Iyy, cv2.CV_64F, cv2.getGaussianKernel(ksize=5, sigma=1))

    #Step 4: Compute a scalar interest measure (R)
    Determinant = Ixx*Iyy - Ixy*Ixy
    Trace = Ixx + Iyy
    R = Determinant - alpha*Trace**2

    #Step 5: Find the local maxima above a threshold and return the results as matrix XYR
    # Define threshold
    threshold = 0.01*np.amax(R)

    # Result matrix
    XYR = []

    for i in range(feature_width, R.shape[0] - feature_width):
        for j in range(feature_width, R.shape[1] - feature_width):
            
            # Find if the point is a local maxima above a threshold
            if R[i,j] > threshold:
                x = int(j)
                y = int(i)
                res_strength = R[i, j]
                # Save results
                XYR.append([x, y, res_strength])

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    
    #############################################################################
    # TODO: YOUR ADAPTIVE NON-MAXIMAL SUPPRESSION CODE HERE                     #
    # While most feature detectors simply look for local maxima in              #
    # the interest function, this can lead to an uneven distribution            #
    # of feature points across the image, e.g., points will be denser           #
    # in regions of higher contrast. To mitigate this problem, Brown,           #
    # Szeliski, and Winder (2005) only detect features that are both            #
    # local maxima and whose response value is significantly (10%)              #
    # greater than that of all of its neighbors within a radius r. The          #
    # goal is to retain only those points that are a maximum in a               #
    # neighborhood of radius r pixels. One way to do so is to sort all          #
    # points by the response strength, from large to small response.            #
    # The first entry in the list is the global maximum, which is not           #
    # suppressed at any radius. Then, we can iterate through the list           #
    # and compute the distance to each interest point ahead of it in            #
    # the list (these are pixels with even greater response strength).          #
    # The minimum of distances to a keypoint's stronger neighbors               #
    # (multiplying these neighbors by >=1.1 to add robustness) is the           #
    # radius within which the current point is a local maximum. We              #
    # call this the suppression radius of this interest point, and we           #
    # save these suppression radii. Finally, we sort the suppression            #
    # radii from large to small, and return the n keypoints                     #
    # associated with the top n suppression radii, in this sorted               #
    # orderself. Feel free to experiment with n, we used n=1500.                #
    #                                                                           #
    # See:                                                                      #
    # https://www.microsoft.com/en-us/research/wp-content/uploads/2005/06/cvpr05.pdf
    # or                                                                        #
    # https://www.cs.ucsb.edu/~holl/pubs/Gauglitz-2011-ICIP.pdf                 #
    #############################################################################

    # Step 1: Sort the XYR matrix by the R column, from largest to smallest R
    XYR = sorted(XYR, key=lambda x: x[2], reverse=True)
    XYR = np.asarray(XYR)

    # Step 2: Iterate through the list and compute the min. distance between each inteest point and all other points ahead
    # x, y values of points
    x = XYR[:,0]
    y = XYR[:,1]

    # Max. number of points
    Nmax = 2500

    # Result matrix
    XYD = []

    for k in range(XYR.shape[0]):
        min_distance = 2*image.shape[0] # maximum possible distance x sqrt(2)

        for kn in range(k):
            dx = x[k] - x[kn]
            dy = y[k] - y[kn]

            # Compute distance between point(k) and neibour point(kn)
            distance = np.sqrt(dx**2 + dy**2)

            # Obtain min. distance 
            if distance < min_distance:
                min_distance = distance

        # Step 3: Save results
        XYD.append([x[k], y[k], min_distance])

    # Step 4: Sort the XYD matrix by the D column (min. distance), from largest to smallest D
    XYD = sorted(XYD, key=lambda x: x[2], reverse=True)
    XYD = np.asarray(XYD)

    # Step 5: Extract the top Nmax points as the final detected points
    XYD = XYD[0:Nmax,:]

    # Return the x, y coordinates of the final detected points
    x = XYD[:,0].astype(int)
    y = XYD[:,1].astype(int)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return x,y, confidences, scales, orientations