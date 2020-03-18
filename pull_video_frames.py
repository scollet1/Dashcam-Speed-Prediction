import sys
import cv2
print(cv2.__version__)

if __name__=="__main__":
	if len(sys.argv) == 2:
		vidcap = cv2.VideoCapture(sys.argv[1])
		success, image = vidcap.read()
		count = 0

		while success:
			cv2.imwrite("data/test/frame%d.jpg" % count, image)
			success, image = vidcap.read()
			print ('Read a new frame: ', success)
			count += 1