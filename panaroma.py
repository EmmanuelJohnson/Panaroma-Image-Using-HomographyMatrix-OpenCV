UBIT = 'emmanueljohnson'
import cv2
import numpy as np
np.random.seed(sum([ord(c) for c in UBIT]))

#References
#https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html

#Basic Configurations
K = 2
RATIO = 0.75
N_MATCHES = 11
MATCH_COLOR = (0,0,255)
RTHRESH = 4.0

#Read the image using opencv
def get_image(path):
    return cv2.imread(path)

#Read the image in gray scale using opencv
def get_image_gray(path):
    return cv2.imread(path,0)

#Show the resulting image
def show_image(name,image):
    cv2.imshow(name,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#Save the resulting image
def save_image(name,image):
    cv2.imwrite(name,image) 

#Extract the keypoints using SIFT
def get_key_points(img):
    sift = cv2.xfeatures2d.SIFT_create()
    kp,ft = sift.detectAndCompute(img, None)
    return kp,ft

#Draw the keypoints and save the result as image
def draw_key_points(img,keypoints,name):
    result = cv2.drawKeypoints(img,keypoints,None)
    save_image(name,result)

#Find the good matches among all the available matches
def get_matches(allMatches):
    matches = []
    ms = []
    for m1, m2 in allMatches:
        change = m2.distance * RATIO
        if m1.distance < change:
            matches.append((m1.trainIdx, m1.queryIdx))
            ms.append(m1)
    return matches, ms

#Getting the corner points of the resulting image
#after transformation
def get_corner_pts(h,w,H):
    cp1 = np.array([[0, 0], [0, h], [w, h], [w, 0]], dtype='float32').reshape(-1, 1, 2)
    cp2 = np.copy(cp1)
    tcp2 = cv2.perspectiveTransform(cp2, H)
    corners = np.concatenate((cp1, tcp2), axis=0)
    return corners

def get_min_pts(h,w,corners):
    cxmin, cymin = np.array(corners.min(axis=0)[0].tolist(), dtype='int32')
    cxmin, cymin = cxmin-1, cymin-1
    cxmax, cymax = np.array(corners.max(axis=0)[0].tolist(), dtype='int32')
    rpanoDim = tuple([cxmax-cxmin, cymax-cymin])
    transH = np.array([[1, 0, -cxmin], [0, 1, -cymin], [0, 0, 1]])
    return transH, rpanoDim, tuple([cxmin,cymin])

#Get random values for a specified range
def get_random_index(r):
    rPicks = np.random.randint(0, r, N_MATCHES)
    return rPicks

#Get only one random value from the specified range
def get_one_random_index(r):
    rPicks = np.random.randint(0, r, 1)
    return rPicks

#Get n random masks
def get_random_matches(masks, limit, max):
    result = np.zeros(max).tolist()
    c = 0
    while True:
        index = get_one_random_index(max)
        if masks[index[0]] == 1:
            result[index[0]] = 1
            c+=1
        if c == limit:
            break
    return result

def main():
    m1 = get_image('mountain1.jpg')
    g1 = get_image_gray('mountain1.jpg')

    m2 = get_image('mountain2.jpg')
    g2 = get_image_gray('mountain2.jpg')

    #Calculate the dimension needed for
    #expanded two images
    pano_width = m1.shape[1]+m2.shape[1]
    pano_height = m1.shape[0]

    #Get keypoints of image1 and image2
    kp1,ft1 = get_key_points(m1)
    kp2,ft2 = get_key_points(m2)

    #Draw the keypoints of image1 and image2
    print('performing task 1.1')
    draw_key_points(g1,kp1,'task1_sift1.jpg')
    draw_key_points(g2,kp2,'task1_sift2.jpg')

    #Use BFMatcher to detect all the matches
    matcher = cv2.BFMatcher()
    allMatches = matcher.knnMatch(ft1, ft2, K)

    #Get the matches whose distance is less than
    #the specified ratio
    matches, ms = get_matches(allMatches)
    
    #using the good matches draw the matches and save the image
    print('performing task 1.2')
    matchimage = cv2.drawMatches(m1, kp1, m2, kp2, ms, None, matchColor=MATCH_COLOR, flags=2)
    save_image('task1_matches_knn.jpg',matchimage)

    keys1,keys2,pts1,pts2 = list(),list(),list(),list()

    for k in kp1:
        keys1.append(k.pt)

    for k in kp2:
        keys2.append(k.pt)

    for m in matches:
        pts1.append(keys1[m[1]])
        pts2.append(keys2[m[0]])

    #Calculate the homography matrix using the 
    #keypoints and using the ransac algorithm
    print('performing task 1.3')
    if len(matches) > 4:
        #H is the homography matrix and mask contains all the points matched
        H, mask = cv2.findHomography(np.asarray(pts1), np.asarray(pts2), cv2.RANSAC, RTHRESH)
    print('\n')
    print('Homography Matrix:')
    print(H)
    print('\n')

    print('performing task 1.4')
    h,w = m1.shape[:2]
    inlierImg = np.zeros((pano_height, pano_width, 3))
    inlierImg[:h, :w] = m1
    inlierImg[:h, w:] = m2
    
    tflist = mask.ravel().tolist()
    npts1, npts2 = list(), list()

    rmatches = get_random_matches(tflist, N_MATCHES, len(tflist))
    rmatches = np.array(rmatches)

    for p1, p2, tf in zip(pts1, pts2, rmatches):
        if tf == 1:
            npts1.append(p1)
            npts2.append(p2)

    rpicksPts1 = np.array(npts1)
    rpicksPts2 = np.array(npts2)

    #Get random points and plot in the image and save it
    zipped = zip(rpicksPts1, rpicksPts2)
    for pt1id, pt2id in zipped:
        pt1 = tuple([int(pt1id[0]),int(pt1id[1])])
        pt2 = tuple([int(pt2id[0]+w),int(pt2id[1])])
        cv2.line(inlierImg, pt1, pt2, MATCH_COLOR, 1)
    save_image('task1_matches.jpg',inlierImg)

    #Warp image1 using Homography Matrix and stitch it with image2
    print('performing task 1.5')
    h,w = m1.shape[:2]
    corners = get_corner_pts(h,w,H)
    transH,rpanoDim,p = get_min_pts(h,w,corners)
    result = cv2.warpPerspective(m1, transH.dot(H), rpanoDim)
    result[-p[1]:-p[1]+h,-p[0]:-p[0]+w] = m2
    save_image('task_pano.jpg',result) 


if __name__ == '__main__':
    main()
