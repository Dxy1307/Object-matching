# Đây là thuật toán so khớp những đặc điểm chính của một đối tượng ở hình ảnh thứ nhất với những mô tả đặc điểm chính trong hình ảnh thứ hai. Mỗi so sánh tạo ra một giá trị khoảng cách và kết quả phù hợp nhất được trả về trên cơ sở khoảng cách nhỏ nhất.

import numpy as np
import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread('img/input/book.png', 0)
img2 = cv2.imread('img/input/books.png', 0)

# initiate ORB detector
orb = cv2.ORB_create()

# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# Tiếp theo, tạo một đối tượng BFMatcher với phép đo khoảng cách cv2.NORM_HAMMING (phù hợp với ORB) và crossCheck được bật để có kết quả tốt hơn. Sau đó, sử dụng phương thức Matcher.match() để có được kết quả phù hợp nhất trong hai hình ảnh. Cần sắp xếp chúng theo thứ tự tăng dần về khoảng cách để xuất hiện những đối tượng phù hợp nhất (với khoảng cách ngắn nhất). Sau đó hiển thị 10 đặc điểm so khớp đầu tiên.

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

# Match descriptors
matches = bf.match(des1, des2)

# sort them in the order of their distance
matches = sorted(matches, key = lambda x:x.distance)

# draw first 10 matches
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags = 2)

plt.imshow(img3)
plt.show()