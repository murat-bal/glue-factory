from ..eval.utils import get_matches_scores
from ..robust_estimators import load_estimator
import cv2
import numpy as np

def compute_homography_SVD(src_points, dst_points):
    """
    Compute the homography matrix H that maps src_points to dst_points.

    Parameters:
    src_points (ndarray): Source points, shape (N, 2)
    dst_points (ndarray): Destination points, shape (N, 2)

    Returns:
    H (ndarray): Homography matrix, shape (3, 3)

    Example usage:
    src_points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])  # Corners of a square
    dst_points = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])  # Transformed square

    H = compute_homography(src_points, dst_points)
    print("Homography matrix:\n", H)
    """
    
    if len(src_points) != len(dst_points):
        raise ValueError("Number of source and destination points must be the same")
    if len(src_points) < 4:
        raise ValueError("At least 4 points are required to compute the homography")

    A = []
    for (x, y), (x_prime, y_prime) in zip(src_points, dst_points):
        # Create the matrix rows for each point correspondence
        A.append([-x, -y, -1, 0, 0, 0, x * x_prime, y * x_prime, x_prime])
        A.append([0, 0, 0, -x, -y, -1, x * y_prime, y * y_prime, y_prime])

    # Convert A to a NumPy array
    A = np.array(A)

    # Solve for H using SVD
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)  # The last row of Vt gives the solution

    # Normalize H so that H[2, 2] = 1
    H = H / H[2, 2]

    return H

def draw_matches(img0, img1, pred):
    
    kp0, kp1 = pred["keypoints0"], pred["keypoints1"]
    m0, scores0 = pred["matches0"], pred["matching_scores0"]
    pts0, pts1, scores = get_matches_scores(kp0, kp1, m0, scores0)
    
    keypoints1_cv = [cv2.KeyPoint(x=int(point[0].item()), y=int(point[1]), size=10) for point in pts0]
    keypoints2_cv = [cv2.KeyPoint(x=int(point[0].item()), y=int(point[1]), size=10) for point in pts1]
    matches = [cv2.DMatch(i, i, 0) for i in range(len(pts0))]
    matched_image = cv2.drawMatches(img0, keypoints1_cv, img1, keypoints2_cv, matches[:3000], (255,255,0,0), cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS, 2)

    cv2.imshow("Matches", matched_image)
    cv2.waitKey(0)

def draw_warped_images(img1, img2, data, pred, conf):
    
    #H_gt = data["H_0to1"]
    kp0, kp1 = pred["keypoints0"], pred["keypoints1"]
    m0, scores0 = pred["matches0"], pred["matching_scores0"]
    pts0, pts1, scores = get_matches_scores(kp0, kp1, m0, scores0)

    results = {}
    
    estimator = load_estimator("homography", conf["estimator"])(conf)

    data_ = {
        "m_kpts0": pts0,
        "m_kpts1": pts1
    }
    est = estimator(data_)

    if est["success"]:
        M = est["M_0to1"]
        inl = est["inliers"].numpy()

        warped_image = warpTwoImages(img2, img1, M.numpy())
        cv2.imshow("Warped Image", warped_image)
        cv2.waitKey(0)
        return warped_image
    
    else :
        return None

def warpTwoImages(img1, img2, H):
    '''warp img2 to img1 with homograph H
    from: https://stackoverflow.com/questions/13063201/how-to-show-the-whole-image-when-using-opencv-warpperspective
    '''
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    pts2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    pts2_ = cv2.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1, pts2_), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin, -ymin]
    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])  # translate

    result = cv2.warpPerspective(img2, Ht@H, (xmax-xmin, ymax-ymin))
    result[t[1]:h1+t[1], t[0]:w1+t[0]] = img1
    return result

def convert_coordinates(w, h , H):  
    # Define the four corners of the image
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]])
    # Convert the corners to homogeneous coordinates
    corners_homogeneous = np.concatenate([corners, np.ones((4, 1))], axis=1)
    print(corners_homogeneous)
    # Apply the homography to the corners
    corners_transformed = np.dot(H, corners_homogeneous.T).T
    # Normalize the transformed corners
    corners_normalized = corners_transformed / corners_transformed[:, 2][:, None]
    return corners_normalized[:, :2]

def showWarpedImages(img_f1, img_f2, H):
    from matplotlib import pyplot as plt
    img1 = cv2.imread(img_f1)
    img2 = cv2.imread(img_f2)
    plt.subplot(2, 2, 1)
    plt.imshow(img1)
    plt.subplot(2, 2, 2)
    plt.imshow(img2)
    warpedImage = warpTwoImages(img1, img2, H)
    warpedImage2 = warpTwoImages(img2, img1, np.linalg.inv(H))
    plt.subplot(2, 2, 3)
    plt.imshow(warpedImage)
    plt.subplot(2, 2, 4)
    plt.imshow(warpedImage2)
    plt.show(block=True)
