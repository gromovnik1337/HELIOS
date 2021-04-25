import numpy as np


def distanceBetweenKeypoints(kpt_1: list, kpt_2: list)-> float:
    """Compute the euclidean distance between two keypoints

    Args:
        kpt_1 (list): keypoint data
        kpt_2 (list): keypoint data

    Returns:
        float: distance
    """
    if kpt_1 == [] or kpt_2 == []:
        return -1

    u = int(kpt_1[0]) - int(kpt_2[0])
    v = int(kpt_1[1]) - int(kpt_2[1])
    vect = (u, v)
    return np.linalg.norm(vect)


def getAngleDefinedByABC(d_b_a: float, d_c_a: float, d_b_c: float) -> float:
    """ Compute the angle theta on the triangle defined by three points A-B-C

                B             
                o------------o C
               / theta
              /
             /  
            /
           o
           A        

    Args:
        d_b_a (float): distance from b to a
        d_c_a (float): distance from c to a
        d_b_c (float): distance from b to c
    Returns:
        float: angle in degrees
        +++
    """
    if d_b_a == -1 or d_c_a == -1 or d_b_c == -1:
        return -1
    theta = np.arccos((d_b_a * d_b_a + d_b_c * d_b_c - d_c_a * d_c_a) / (2 * d_b_a * d_b_c))
    return theta * 180 / np.pi


def toDicitonary(keypoints: list)->dict:
    """Convert the list of the keypoint data into a dictionary

    Args:
        keypoints (list): Data for all the keypoints

    Returns:
        dict: dictionary containing the keypoint data
    """
    keypoints_data = {}

    for i in range(len(keypoints)):
        if keypoints[i] != []:
            k = keypoints[i][0]
        else:
            k = []
        # Build a dictionary to save the keypoint data
        keypoints_data.update({'K_' + str(i): k})
    return keypoints_data


def getKeypointsData(keypoints):
    keypoints_data = toDicitonary(keypoints)
    # Compute the distance from the left elbow to the left shoulder
    keypoints_data.update({'d_le_ls': distanceBetweenKeypoints(keypoints_data['K_6'], keypoints_data['K_5'])})
    # Compute the distance from the right elbow to the right shoulder
    keypoints_data.update({'d_re_rs':distanceBetweenKeypoints(keypoints_data['K_3'], keypoints_data['K_2'])})
    # Compute the distance from the right elbow to the right wrist
    keypoints_data.update({'d_re_rw':distanceBetweenKeypoints(keypoints_data['K_3'],keypoints_data['K_4'])})
    # Compute the distance from the left elbow to the left wrist
    keypoints_data.update({'d_le_lw':distanceBetweenKeypoints(keypoints_data['K_6'],keypoints_data['K_7'])})
    # Compute the distance from the left shoulder to the left wrist
    keypoints_data.update({'d_ls_lw':distanceBetweenKeypoints(keypoints_data['K_5'],keypoints_data['K_7'])})
    # Compute the distance from the right shoulder to the right wrist
    keypoints_data.update({'d_rs_rw':distanceBetweenKeypoints(keypoints_data['K_2'],keypoints_data['K_4'])})
    # Compute the distance from the right shoulder to the left shoulder
    keypoints_data.update({'d_rs_ls':distanceBetweenKeypoints(keypoints_data['K_5'],keypoints_data['K_2'])})
    # Angle left elbow
    keypoints_data.update({'theta_le':getAngleDefinedByABC(keypoints_data['d_le_lw'], keypoints_data['d_ls_lw'], keypoints_data['d_le_ls'])})
    # Angle right elbow 
    keypoints_data.update({'theta_re':getAngleDefinedByABC(keypoints_data['d_re_rw'], keypoints_data['d_rs_rw'], keypoints_data['d_re_rs'])})

    return keypoints_data
