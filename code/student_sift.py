import numpy as np
import cv2
from scipy import signal
import math
import pandas as pd
import sys

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
    # decrease your matching accuracy.

    #Hyper Parameter
    GuassianKernelSize = 5
    normalization = True

    #Create Gaussian blurred version of original image
    a = cv2.getGaussianKernel(GuassianKernelSize, 1) * cv2.getGaussianKernel(GuassianKernelSize, 1).T
    image_GaussianBlurred = signal.convolve(image,a)
    image_GaussianBlurred = np.delete(image_GaussianBlurred, -1, 0)
    image_GaussianBlurred = np.delete(image_GaussianBlurred, 0, 0)
    image_GaussianBlurred = np.delete(image_GaussianBlurred, -1, 1)
    image_GaussianBlurred = np.delete(image_GaussianBlurred, 0, 1)
    np.set_printoptions(threshold=np.inf)

    def Extract_Magnitude(yy,xx):
        return math.sqrt(((image_GaussianBlurred[xx+1,yy]-image_GaussianBlurred[xx-1,yy])**2)+((image_GaussianBlurred[xx,yy+1]-image_GaussianBlurred[xx,yy-1])**2))

    def Extract_Direction(yy,xx):
        #8-directional catagorization
        #↑: 0, ↗: 1, →: 2, ↘: 3, ↓: 4, ↙: 5, ←:  6, ↖: 7
        #degrees=math.degrees(math.atan((image_GaussianBlurred[xx, yy+1] - image_GaussianBlurred[xx , yy-1])/(image_GaussianBlurred[xx+1, yy] - image_GaussianBlurred[xx-1 , yy])))
        degrees=math.degrees(math.atan2((image_GaussianBlurred[xx, yy+1] - image_GaussianBlurred[xx , yy-1]),(image_GaussianBlurred[xx+1, yy] - image_GaussianBlurred[xx-1 , yy])))

        degrees = degrees%360
        #print(degrees)
        a= int((degrees +22.5)//45)

        if a == 8:
            return int(0)
        else:
            return a

    y = np.reshape(y,(y.shape[0] ,1))
    x = np.reshape(x, (x.shape[0], 1))

    integrated = np.concatenate((x,y),axis=1)
    binned = np.zeros([integrated.shape[0],128])

    ######
    #for order in range(2, 3):
    for order in range (0, integrated.shape[0]):

        sys.stdout.write('\r' + "   " + str(int(order / integrated.shape[0] * 100 + 1)) + "%")
        x_center = math.floor(integrated[order,0])+0.5
        y_center = math.floor(integrated[order, 1]) + 0.5
        dummy = np.zeros([feature_width,feature_width,2])

        for x2 in range (0,feature_width):
            for y2 in range(0, feature_width):


                x_co = feature_width/(-2)+0.5+x2
                y_co = feature_width / (-2) + 0.5 + y2
                dummy[int(feature_width/(2)-0.5+x_co),int(feature_width/(2)-0.5+y_co),0] = Extract_Magnitude(int(x_center+x_co),int(y_center+y_co))
                dummy[int(feature_width/(2)-0.5+x_co),int(feature_width/(2)-0.5+y_co), 1] = Extract_Direction(int(x_center+x_co),int(y_center+y_co))

        #Binned part generation
        #Cell movement
        for  x_co in range (0,4):
            for y_co in range(0, 4):
                #mini-Cell movement
                for x_sco in range(0, 4):
                    for y_sco in range(0, 4):
                        binned[order,int(((x_co)*4+y_co)*8+dummy[4*x_co+x_sco,4*y_co+y_sco,1])] += dummy[4*x_co+x_sco,4*y_co+y_sco,0]
    #df = pd.DataFrame(dummy[:,:,0])
    #df2 = pd.DataFrame(dummy[:,:,1])
    #df3 = pd.DataFrame(binned)

    #df.to_excel('dummy1.xlsx', index=False)
    #df2.to_excel('dummy2.xlsx', index=False)
    #df3.to_excel('binned_s.xlsx', index=False)
    print()


    #Binned part normalization
    if normalization ==True:
        for order in range (0,integrated.shape[0]):
            for seq in range (0, feature_width):
                temp_sum = 0
                for dir_order in range (0,8):
                    temp_sum += binned[order,(seq*8)+dir_order]
                for dir_order in range (0,8):
                    binned[order, (seq * 8) + dir_order] =  binned[order,(seq*8)+dir_order]/temp_sum
                #print("temp sum", temp_sum)

    #df4.to_excel('binned_normalized.xlsx', index=False)




    #############################################################################

    #raise NotImplementedError('`get_features` function in ' +
    #    '`student_sift.py` needs to be implemented')

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return binned



