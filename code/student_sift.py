import numpy as np
import cv2


def get_features(image, x, y, feature_width, scales=None):
    """
    To start with, you might want to simply use normalized patches as your
    local feature. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT descriptor
    (See Szeliski 4.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each feature_width/4. It is simply the
        terminology used in the feature literature to describe the spatial
        bins where gradient distributions will be described.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length.

    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.

    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like feature can be constructed entirely
    from filtering fairly quickly in this way.

    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.

    Args:
    -   image: A numpy array of shape (m,n) or (m,n,c). can be grayscale or color, your choice
    -   x: A numpy array of shape (k,), the x-coordinates of interest points
    -   y: A numpy array of shape (k,), the y-coordinates of interest points
    -   feature_width: integer representing the local feature width in pixels.
            You can assume that feature_width will be a multiple of 4 (i.e. every
                cell of your local SIFT-like feature will have an integer width
                and height). This is the initial window size we examine around
                each keypoint.
    -   scales: Python list or tuple if you want to detect and describe features
            at multiple scales

    You may also detect and describe features at particular orientations.

    Returns:
    -   fv: A numpy array of shape (k, feat_dim) representing a feature vector.
            "feat_dim" is the feature_dimensionality (e.g. 128 for standard SIFT).
            These are the computed features.
    """
    assert image.ndim == 2, 'Image must be grayscale'
    #############################################################################
    # TODO: YOUR CODE HERE                                                      #
    # If you choose to implement rotation invariance, enabling it should not    #
    # decrease your matching accuracy.                                          #
    #############################################################################

    # Step 1: Compute the horizontal and vertical derivatives of the image (Ix, Iy) 
    Ix = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Step 2: Compute magnitudes (I_magnitudes) and angles (I_angles) and of the image
    I_magnitudes = np.sqrt(Ix ** 2 + Iy ** 2)
    I_angles = np.arctan2(Iy, Ix)

    # Result matrix
    fv = []

    # Step 3: Iterate through interest points(k) 
    for k in range(x.shape[0]):
        # Extract the 16x16 window (point(k) is in the center of this window) 
        xc = int(x[k])
        yc = int(y[k])

        fw_by2 = int(feature_width/2)
        x_start = max(xc - fw_by2, 0)
        x_end = min(xc + fw_by2, image.shape[1])
        y_start = max(yc - fw_by2, 0)
        y_end = min(yc + fw_by2, image.shape[0])

        # Obtain magnitudesWin16x16) & anglesWin16x16
        magnitudesWin16x16 = I_magnitudes[y_start:y_end, x_start:x_end]
        anglesWin16x16 = I_angles[y_start:y_end, x_start:x_end]
        
        # Result matrix
        siftFeatures_vector = []

        # Step 4: Divide the 16x16 windows to 16*4x4 windows (magnitudesWin4x4 , anglesWin4x4);
        fw_by4 = int(feature_width/4)
        for i in range(0, feature_width, 4):
            for j in range(0, feature_width, 4):
                # Obtain magnitudesWin4x4 & anglesWin4x4
                magnitudesWin4x4 = magnitudesWin16x16[i:i + fw_by4, j:j + fw_by4]
                anglesWin4x4 = anglesWin16x16[i:i + fw_by4, j:j + fw_by4]
                
                # Compute 8-bin histogram of anglesWin4x4 weighted by magnitudesWin4x4 
                hist_vector, _ = np.histogram(anglesWin4x4, bins=8, range=(-np.pi, np.pi), weights=magnitudesWin4x4)
                siftFeatures_vector.extend(hist_vector)

        # Step 5: Arrange and save 16 hist_vector as the siftFeatures_vector of point(k)
        siftFeatures_vector = np.array(siftFeatures_vector)

        # Normalize siftFeatures_vector;
        siftFeatures_vector = siftFeatures_vector/np.sqrt(np.sum(siftFeatures_vector**2))

        # Arrange and save N siftFeatures_vector as fv matrix 
        fv.append(siftFeatures_vector)

    fv = np.array(fv)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return fv
