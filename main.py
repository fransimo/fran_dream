import os
import cv2

# https://www.codespeedy.com/find-a-specific-object-in-an-image-using-opencv-in-python/
# https://docs.opencv.org/3.1.0/d7/d8b/tutorial_py_face_detection.html
# http://svn.fransimo.info/fransimo/WANA_oCVFinder/trunk/WANA_oCVFinder.cpp
# http://svn.fransimo.info/fransimo/WANA_oCVTrain/aptget_install.sh

# https://github.com/opencv/opencv/blob/4.1.0/samples/python/facedetect.py#L22
# https://github.com/opencv/opencv/tree/4.1.0/samples/python

# https://opencv-tutorial.readthedocs.io/en/latest/face/face.html


def search_in_file(detector, file_path, output_path):
    splited = output_path.split('.')
    output_path = splited[0] + '_{:04d}_{:04d}.' + splited[1]

    img = cv2.imread(file_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray = cv2.equalizeHist(gray)

    # http://svn.fransimo.info/fransimo/WANA_oCVFinder/trunk/WANA_oCVFinder.cpp
    # 	// There can be more than one face in an image. So create a growable sequence of faces.
    # 		// Detect the objects and store them in the sequence
    # 		CvSeq* faces = cvHaarDetectObjects(img, cascade, storage, 1.1, 2,
    # 				CV_HAAR_DO_CANNY_PRUNING, cvSize(40, 40));
    # https://github.com/opencv/opencv/issues/11716

    faces = detector.detectMultiScale(gray, scaleFactor=1.1,
                                      minNeighbors=2,
                                      minSize=(40, 40),
                                      flags=cv2.CASCADE_DO_CANNY_PRUNING)

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            dsize = (3000, 3000)

            roi_color = img[y:y + h, x:x + w]
            # https://pythonexamples.org/python-opencv-cv2-resize-image/
            roi_color = cv2.resize(roi_color, dsize, interpolation=cv2.INTER_LINEAR)

            output_path_w = output_path.format(x, y)
            try:
                cv2.imwrite(output_path_w, roi_color)
            except:
                print('error on:' + output_path_w)


def search(input_directory='.', output_directory=None):
    if not output_directory:
        output_directory = input_directory + '/out'

    detector = cv2.CascadeClassifier('haarcascade_u10.10_7_v2.xml')
    i = 0
    for filename in os.listdir(input_directory):
        if filename.endswith(".jpg"):
            search_in_file(detector, input_directory + '/' + filename, output_directory + '/' + filename)
            i = i + 1
        if i > 10000:
            break


if __name__ == '__main__':
    search('/Users/fran/Pictures/salida/tagged')


