import cv2

	
image = cv2.imread('./test.jpg')
	
    
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# cv2.imshow('Original image',image)
# cv2.imshow('Gray image', gray)

cv2.imwrite('./Test_gray.jpg', image_gray) 