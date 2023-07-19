import cv2
import numpy as np
import utlis

###############################################
webCamFeed = False
pathImage = "Resources/shapes.jpg"
cap = cv2.VideoCapture("Resources/test_video.mp4")
cap.set(10,160)
heightImg = 640
widthImg  = 480
###############################################

### FOR VIDEO ###
if not webCamFeed:
    success, img = cap.read()
else:
    img = cv2.imread(pathImage)
img = cv2.resize(img, (widthImg, heightImg))
imgContours = img.copy()
imgBigContour = img.copy()

# Initialize object tracker
tracker = cv2.TrackerKCF_create()

# Initialize variables for page tracking
is_tracking = False
last_center = None
frame_counter = 0
page_counter = 0
turn_threshold = 50  # or any other value that you want to use as the turn threshold


while True:
    ### FOR VIDEO ###
    if not webCamFeed:
        success, img = cap.read()
    else:
        img = cv2.imread(pathImage)

    img = cv2.resize(img, (widthImg, heightImg))
    imgFinal = img.copy()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1.5)
    imgCanny = cv2.Canny(imgBlur, 50, 50)
    kernel = np.ones((5, 5))
    imgDil = cv2.dilate(imgCanny, kernel, iterations=1)
    imgEro = cv2.erode(imgDil, kernel, iterations=1)

    contours, hierarchy = cv2.findContours(imgEro, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)

    # FIND THE BIGGEST CONTOUR
    biggest, maxArea = utlis.biggestContour(contours)
    if biggest.size != 0:
        biggest = utlis.reorder(biggest)
        cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20) # DRAW THE BIGGEST CONTOUR
        imgBigContour = utlis.drawRectangle(imgBigContour,biggest,2)
        pts1 = np.float32(biggest) # PREPARE POINTS FOR WARP
        pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

        # REMOVE 20 PIXELS FROM EACH SIDE
        imgWarpColored = imgWarpColored[20:imgWarpColored.shape[0] - 20, 20:imgWarpColored.shape[1] - 20]


        
        # Check if page is being tracked
        if is_tracking:
            # Calculate difference between current frame and previous tracked frame
            diff = cv2.absdiff(imgWarpColored, prev_frame)

            # Calculate mean intensity value of difference image
            mean_intensity = np.mean(diff)

            # If mean intensity is below threshold, consider page to be turned
            if mean_intensity < turn_threshold:
                # Reset tracking variables
                is_tracking = False
                prev_frame = None
                frame_num += 1

                # Save image of new page
                cv2.imwrite(f"page_{frame_num}.jpg", imgWarpColored)

            else:
                # Update previous frame with current frame
                prev_frame = imgWarpColored.copy()

        # If page is not being tracked, check for page turn
        else:
            # Calculate difference between current frame and previous frame
            if prev_frame is not None:
                diff = cv2.absdiff(imgWarpColored, prev_frame)

                # Calculate mean intensity value of difference image
                mean_intensity = np.mean(diff)

                # If mean intensity is above threshold, start tracking new page
                if mean_intensity > start_tracking_threshold:
                    is_tracking = True
                    prev_frame = imgWarpColored.copy()

            # Update previous frame with current frame
            prev_frame = imgWarpColored.copy()

            # Display image
            cv2.imshow("Result", imgWarpColored)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

