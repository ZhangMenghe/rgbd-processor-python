import numpy as np
from scipy import signal
import cv2
from utils import invertIt,mutiplyIt,getPointCloudFromZ,getRMatrix,rotatePC,getYDir
def filterItChopOff(f, r, sp):
    f[np.isnan(f)] = 0
    H, W, d = f.shape
    B = np.ones([2 * r + 1, 2 * r + 1])

    minSP = cv2.erode(sp, B, iterations=1)
    maxSP = cv2.dilate(sp, B, iterations=1)

    ind = np.where(np.logical_or(minSP != sp, maxSP != sp))

    spInd = np.reshape(range(np.size(sp)), sp.shape,'F')

    delta = np.zeros(f.shape)
    delta = np.reshape(delta, (H * W, d), 'F')
    f = np.reshape(f, (H * W, d),'F')

    # calculate delta

    I, J = np.unravel_index(ind, [H, W], 'C')
    for i in range(np.size(ind)):
        x = I[i]
        y = J[i]
        clipInd = spInd[max(0, x - r):min(H-1, x + r), max(0, y - r):min(W-1, y + r)]
        diffInd = clipInd[sp[clipInd] != sp[x, y]]
        delta[ind[i], :] = np.sum(f[diffInd, :], 1)
    delta = np.reshape(delta, (H, W, d), 'F')
    f = np.reshape(f, (H, W, d), 'F')
    fFilt = np.zeros([H, W, d])

    for i in range(f.shape[2]):
        #  fFilt(:,:,i) = filter2(B, f(:,:,i));
        tmp = signal.convolve2d(np.rot90(f[:, :, i], 2), np.rot90(np.rot90(B, 2), 2), mode="same")
        fFilt[:, :, i] = np.rot90(tmp, 2)
    fFilt = fFilt - delta
    return fFilt

'''
Clip out a 2R+1 x 2R+1 window at each point and estimate
the normal from points within this window. In case the window
straddles more than a single superpixel, only take points in the
same superpixel as the centre pixel. Takes about 0.5 second per image.
Does not use ImageStack so is more platform independent, but
gives different results.
Input:
missingMask:  boolean mask of what data was missing
pc:           X,Y,Z coordinates in real world
R:            radius of clipping
sc:           to upsample or not
superpixels:  superpixel map to define bounadaries that should
not be straddled

Output: The normal at pixel (x,y) is N(x, y, :)'pt + b(x,y) = 0
N:            Normal field
b:            bias
'''
def computeNormalsSquareSupport(missingMask, pc, R, superpixels):
    XYZf = np.copy(pc)
    XYZ = np.copy(pc)
    X, Y, Z = XYZ[:,:,0],XYZ[:,:,1],XYZ[:,:,2]
    ind = np.where(missingMask == 1)
    X[ind] = np.nan
    Y[ind] = np.nan
    Z[ind] = np.nan

    one_Z = np.expand_dims(1 / Z, axis=2)
    X_Z = np.divide(X, Z)
    Y_Z = np.divide(Y, Z)
    one = np.copy(Z)
    one[np.invert(np.isnan(one[:, :]))] = 1
    ZZ = np.multiply(Z, Z)
    X_ZZ = np.expand_dims(np.divide(X, ZZ), axis=2)
    Y_ZZ = np.expand_dims(np.divide(Y, ZZ), axis=2)

    X_Z_2 = np.expand_dims(np.multiply(X_Z, X_Z), axis=2)
    XY_Z = np.expand_dims(np.multiply(X_Z, Y_Z), axis=2)
    Y_Z_2 = np.expand_dims(np.multiply(Y_Z, Y_Z), axis=2)

    AtARaw = np.concatenate((X_Z_2, XY_Z, np.expand_dims(X_Z, axis=2), Y_Z_2,
                             np.expand_dims(Y_Z, axis=2), np.expand_dims(one, axis=2)), axis=2)

    AtbRaw = np.concatenate((X_ZZ, Y_ZZ, one_Z), axis=2)

    # with clipping
    AtA = filterItChopOff(np.concatenate((AtARaw, AtbRaw), axis=2), R, superpixels)
    Atb = AtA[:, :, AtARaw.shape[2]:]
    AtA = AtA[:, :, :AtARaw.shape[2]]

    AtA_1, detAtA = invertIt(AtA)
    N = mutiplyIt(AtA_1, Atb)

    divide_fac = np.sqrt(np.sum(np.multiply(N, N), axis=2))
    # with np.errstate(divide='ignore'):
    b = np.divide(-detAtA, divide_fac)
    for i in range(3):
        N[:, :, i] = np.divide(N[:, :, i], divide_fac)

        # Reorient the normals to point out from the scene.
    # with np.errstate(invalid='ignore'):
    SN = np.sign(N[:, :, 2])
    SN[SN == 0] = 1
    extend_SN = np.expand_dims(SN, axis=2)
    extend_SN = np.concatenate((extend_SN, extend_SN, extend_SN), axis=2)
    N = np.multiply(N, extend_SN)
    b = np.multiply(b, SN)
    sn = np.sign(np.sum(np.multiply(N, XYZf), axis=2))
    sn[np.isnan(sn)] = 1
    sn[sn == 0] = 1
    extend_sn = np.expand_dims(sn, axis=2)
    N = np.multiply(extend_sn, N)
    b = np.multiply(b, sn)
    return N, b


'''
processDepthImage
Input: z value in cm, C is intrinsic camera matrix
Output: pc : height x width x 3 , XYZ in real world
        N : height x width x 3 normal vector for each pixel
        yDir: 3x1 y direction(note it is the opposite dir)
        h: height x width height (could be negtive if under camera)for each pixel
        R: 3x3 correction rotation matrix
'''
def processDepthImage(depthImage, missingMask, C):
    ydir_angleThresh = np.array([45, 15])
    ydir_iter = np.array([5, 5])
    ydir_y0 = np.array([[0, 1, 0]]).T

    normal_patch_size = np.array([3, 10])

    pc = np.zeros([depthImage.shape[0], depthImage.shape[1], 3])
    pc[:, :, 0], pc[:, :, 1], pc[:, :, 2] = getPointCloudFromZ(depthImage, C, 1)
    N1, b1 = computeNormalsSquareSupport(missingMask, pc, normal_patch_size[0], np.ones(depthImage.shape))
    N2, b2 = computeNormalsSquareSupport(missingMask, pc, normal_patch_size[1], np.ones(depthImage.shape))

    N = N1
    yDir = getYDir(N2, ydir_angleThresh, ydir_iter, ydir_y0)
    y0 = np.array([[0, 1, 0]]).T
    R = getRMatrix(y0, yDir)
    pcRot = rotatePC(pc, R.T)
    h = -pcRot[:,:,1]
    yMin = np.percentile(h, 0)
    if (yMin > -90):
        yMin = -130
    h = h - yMin
    return pc, N, yDir, h, R.T
