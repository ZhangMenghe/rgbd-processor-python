import numpy as np
import os
def checkDirAndCreate(folder, checkNameList = ['hha','height']):
    if not os.path.exists(folder):
        try:
            os.makedirs(folder)
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    for folderName in checkNameList:
        fullName = folder + '/' + folderName + '/'
        if not os.path.exists(fullName):
            try:
                os.makedirs(fullName)
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

'''
Input: AtA: W x H x 6
'''

def invertIt(AtA):
    AtA_1 = np.zeros([AtA.shape[0], AtA.shape[1], 6])
    AtA_1[:, :, 0] = np.multiply(AtA[:, :, 3], AtA[:, :, 5]) - np.multiply(AtA[:, :, 4], AtA[:, :, 4])
    AtA_1[:, :, 1] = -np.multiply(AtA[:, :, 1], AtA[:, :, 5]) + np.multiply(AtA[:, :, 2], AtA[:, :, 4])
    AtA_1[:, :, 2] = np.multiply(AtA[:, :, 1], AtA[:, :, 4]) - np.multiply(AtA[:, :, 2], AtA[:, :, 3])
    AtA_1[:, :, 3] = np.multiply(AtA[:, :, 0], AtA[:, :, 5]) - np.multiply(AtA[:, :, 2], AtA[:, :, 2])
    AtA_1[:, :, 4] = -np.multiply(AtA[:, :, 0], AtA[:, :, 4]) + np.multiply(AtA[:, :, 1], AtA[:, :, 2])
    AtA_1[:, :, 5] = np.multiply(AtA[:, :, 0], AtA[:, :, 3]) - np.multiply(AtA[:, :, 1], AtA[:, :, 1])

    x1 = np.multiply(AtA[:, :, 0], AtA_1[:, :, 0])
    x2 = np.multiply(AtA[:, :, 1], AtA_1[:, :, 1])
    x3 = np.multiply(AtA[:, :, 2], AtA_1[:, :, 2])

    detAta = x1 + x2 + x3
    return AtA_1, detAta


'''
mutiplyIt
'''


def mutiplyIt(AtA_1, Atb):
    result = np.zeros([Atb.shape[0], Atb.shape[1], 3])
    result[:, :, 0] = np.multiply(AtA_1[:, :, 0], Atb[:, :, 0]) + np.multiply(AtA_1[:, :, 1],
                                                                              Atb[:, :, 1]) + np.multiply(
        AtA_1[:, :, 2], Atb[:, :, 2])
    result[:, :, 1] = np.multiply(AtA_1[:, :, 1], Atb[:, :, 0]) + np.multiply(AtA_1[:, :, 3],
                                                                              Atb[:, :, 1]) + np.multiply(
        AtA_1[:, :, 4], Atb[:, :, 2])
    result[:, :, 2] = np.multiply(AtA_1[:, :, 2], Atb[:, :, 0]) + np.multiply(AtA_1[:, :, 4],
                                                                              Atb[:, :, 1]) + np.multiply(
        AtA_1[:, :, 5], Atb[:, :, 2])
    return result
'''
getPointCloudFromZ: get position in real world
Input: Z is in cm
       C: camera matrix
       s: factor by which Zhas ben upsampled
'''
def getPointCloudFromZ(Z,C,s = 1):
    height, width= Z.shape
    xx,yy = np.meshgrid(np.array(range(width))+1, np.array(range(height))+1)
    # color camera parameters
    cc_rgb = C[0:2,2] * s
    fc_rgb = np.diag(C[0:2,0:2]) * s
    x3 = np.multiply((xx - cc_rgb[0]), Z) / fc_rgb[0]
    y3 = np.multiply((yy - cc_rgb[1]), Z) / fc_rgb[1]
    z3 = Z
    return x3, y3, z3
'''
getYDir: get gravity direction
Input: N: HxWx3 normal field
       angleThresh: degrees the threshold for mapping to parallel to gravity and perpendicular to gravity
       num_iter: number of iterations
       y0: initial gravity direction
Output:gravity direction
'''
def getYDirHelper(N, y0, thresh, num_iter):
    dim = N.shape[0] * N.shape[1]
    nn = np.swapaxes(np.swapaxes(N,0,2),1,2)
    nn = np.reshape(nn, (3, dim), 'F')
    # only keep those non-nan
    idx = np.where(np.invert(np.isnan(nn[0,:])))[0]
    nn = nn[:,idx]
    yDir = y0;
    for i in range(num_iter):
        sim0 = np.dot(yDir.T, nn)
        indF = abs(sim0) > np.cos(thresh)
        indW = abs(sim0) < np.sin(thresh)
        if(len(indF.shape) == 2):
            NF = nn[:, indF[0,:]]
            NW = nn[:, indW[0,:]]
        else:
            NF = nn[:, indF]
            NW = nn[:, indW]
        A = np.dot(NW, NW.T) - np.dot(NF, NF.T)
        b = np.zeros([3,1])
        c = NF.shape[1]
        w,v = np.linalg.eig(A)
        min_ind = np.argmin(w)
        newYDir = v[:,min_ind]
        yDir = newYDir * np.sign(np.dot(yDir.T, newYDir))
    return yDir
def getYDir(N, angleThresh, num_iter, y0):
    y = y0
    for i in range(len(angleThresh)):
        thresh = np.pi*angleThresh[i]/180
        y = getYDirHelper(N, y, thresh, num_iter[i])
    return y


'''
getRMatrix: Generate a rotation matrix that
            if yf is a scalar, rotates about axis yi by yf degrees
            if yf is an axis, rotates yi to yf in the direction given by yi x yf
Input: yi is an axis 3x1 vector
       yf could be a scalar of axis

'''


def getRMatrix(yi, yf):
    if (np.isscalar(yf)):
        ax = yi / np.linalg.norm(yi)
        phi = yf
    else:
        yi = yi / np.linalg.norm(yi)
        yf = yf / np.linalg.norm(yf)
        ax = np.cross(yi.T, yf.T).T
        ax = ax / np.linalg.norm(ax)
        # find angle of rotation
        phi = np.degrees(np.arccos(np.dot(yi.T, yf)))

    if (abs(phi) > 0.1):
        phi = phi * (np.pi / 180)

        s_hat = np.array([[0, -ax[2], ax[1]],
                          [ax[2], 0, -ax[0]],
                          [-ax[1], ax[0], 0]])
        R = np.eye(3) + np.sin(phi) * s_hat + (1 - np.cos(phi)) * np.dot(s_hat, s_hat)
    else:
        R = np.eye(3)
    return R

def rotatePC(pc,R):
    if( np.array_equal(R, np.eye(3))):
        return pc
    else:
        dim = pc.shape[0] * pc.shape[1]
        pc = np.swapaxes(np.swapaxes(pc, 0, 2), 1, 2)
        res = np.reshape(pc, (3, dim), 'F')
        res = np.dot(R, res)
        res = np.reshape(res, pc.shape, 'F')
        res = np.swapaxes(np.swapaxes(res, 0, 1), 1, 2)
        return res
