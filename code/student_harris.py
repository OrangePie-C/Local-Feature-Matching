import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

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
    # TODO: YOUR HARRIS CORNER DETECTOR CODE HERE
    #
    #Hyper parameter
    k=0.2
    threshold = 0.05
    suppression_kernal_size = 6
    density = 8
    interval = int(feature_width/density)
    R_map = np.empty_like(image)

    y_gradient_map, x_gradient_map = np.gradient(image)

    x_integrated = np.array([])
    y_integrated = np.array([])
    R_integrated = np.array([])

    x_gradient_map = np.square(x_gradient_map)
    y_gradient_map = np.square(y_gradient_map)
    xy_gradient_map = x_gradient_map * y_gradient_map

    #Window Correction(make window size as (odd#)x(odd#): selective enlargement
    new_window_size = int(round(feature_width/2,0)*2+1)
    approx_half_window = int((new_window_size - 1) / 2)

    print("   R-Value Generation, and x/y coordinate localization_start")
    #R-value generation
    for x in range(approx_half_window+int(suppression_kernal_size/2),image.shape[0]-(approx_half_window)-int(suppression_kernal_size/2),interval):
        sys.stdout.write('\r' +"   "+ str(int((x+8) / image.shape[0] * 100 + 1)) + "%")
        for y in range((approx_half_window+int(suppression_kernal_size/2)),image.shape[1]-(approx_half_window)-int(suppression_kernal_size/2),interval):
            Ix_sqrd_Sum = 0
            Iy_sqrd_Sum = 0
            Ixy_sqrd_Sum = 0
            for x2 in range(-1*(approx_half_window),(approx_half_window+1)):
                for y2 in range(-1*approx_half_window,(approx_half_window+1)):
                    Ix_sqrd_Sum = Ix_sqrd_Sum + x_gradient_map[x-x2][y-y2]
                    Iy_sqrd_Sum = Iy_sqrd_Sum + y_gradient_map[x - x2][y - y2]
                    Ixy_sqrd_Sum = Ixy_sqrd_Sum + xy_gradient_map[x - x2][y - y2]
            det = (Ix_sqrd_Sum*Iy_sqrd_Sum) - (Ixy_sqrd_Sum*Ixy_sqrd_Sum)
            trace = 2*Ixy_sqrd_Sum
            R= det - (k*trace*trace)
            R_map[x, y] = R
            if R > threshold:
                x_integrated = np.append(x_integrated,x)
                y_integrated = np.append(y_integrated,y)
                R_integrated = np.append(R_integrated,R)


    #Local Minima Suppression
        #normal suppression kernal
    print()
    print("   Non-Maxima Suppression_start")
    suppression_kernal = np.ones([suppression_kernal_size, suppression_kernal_size])
        #normal round kernal
    #suppresion_kernal = ???
    for x in range(int(suppression_kernal_size / 2), R_map.shape[0] - int(suppression_kernal_size / 2)):
        #print((x-int(suppression_kernal_size / 2))/R_map.shape[0])
        for y in range(int(suppression_kernal_size / 2), R_map.shape[1] - int(suppression_kernal_size / 2)):
            if R_map[x,y]>threshold:
                dummy = np.empty_like(suppression_kernal)
                for xsk in range(0, suppression_kernal_size):
                    for ysk in range(0, suppression_kernal_size):
                        dummy[xsk, ysk] = suppression_kernal[xsk, ysk] * R_map[
                            x + (xsk - int(suppression_kernal_size / 2)), y + (ysk - int(suppression_kernal_size / 2))]
                for xd in range(0, dummy.shape[0]):
                    for yd in range(0, dummy.shape[1]):
                        if dummy[xd, yd] < dummy.max():
                            R_map[x + (xd - int(dummy.shape[0] / 2)), y + (yd - int(dummy.shape[1] / 2))] = 0

    x_integrated = np.array([])
    y_integrated = np.array([])
    R_integrated = np.array([])

    for x in range (0, R_map.shape[0]):
        for y in range(0, R_map.shape[1]):
            if R_map[x,y] > threshold:
                x_integrated = np.append(x_integrated,x)
                y_integrated = np.append(y_integrated, y)


    # For this algorithm, quadratic approximation is excluded
    # because feature description part already approximates the x and y values
    # to be having ~.5 values to be center of 16x16 size patch.


    #############################################################################
    '''
    raise NotImplementedError('`get_interest_points` function in ' +
    '`student_harris.py` needs to be implemented')
    '''
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    
    #############################################################################
    # TODO: YOUR ADAPTIVE NON-MAXIMAL SUPPRESSION CODE HERE (for extra credit)  #
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
    '''
    raise NotImplementedError('adaptive non-maximal suppression in ' +
    '`student_harris.py` needs to be implemented')
    '''
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    #return x,y, confidences, scales, orientations
    return x_integrated, y_integrated


