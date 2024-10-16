# import cv2
# import numpy as np
#
# # Mở camera
# cap = cv2.VideoCapture(1)  # '0' là camera mặc định, nếu bạn sử dụng camera ngoài, có thể thay thành 1 hoặc 2
#
# # Vòng lặp để liên tục đọc khung hình từ camera
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Không thể truy cập camera.")
#         break
#
#     # Resize khung hình cho dễ xử lý
#     frame = cv2.resize(frame, (800, 800))
#
#     # Chuyển đổi khung hình sang thang màu xám
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     # Làm mờ ảnh để giảm nhiễu
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#
#     # Phát hiện cạnh bằng phương pháp Canny (phát hiện cạnh)
#     edges = cv2.Canny(blurred, 100, 300)
#
#     # Tìm các đường viền (contours)
#     contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     #vẽ lên khung hình gôốc
#     cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)  # Vẽ tất cả các contours lên khung hình gốc
#
#     # Vẽ các đường viền và nhận diện hình chữ nhật và hình tròn
#     for contour in contours:
#         approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
#
#         # Nhận diện hình chữ nhật là bảng rổ
#         if len(approx) == 4:
#             cv2.drawContours(frame, [approx], 0, (0, 255, 0), 5)
#
#         # Nhận diện hình tròn là vành rổ
#         area = cv2.contourArea(contour)
#         if area > 500:  # Lọc các vật thể nhỏ hơn
#             (x, y), radius = cv2.minEnclosingCircle(contour)
#             if radius > 30:  # Giới hạn kích thước của vành rổ
#                 cv2.circle(frame, (int(x), int(y)), int(radius), (0, 0, 255), 5)
#
#     # Hiển thị khung hình đã qua xử lý
#     cv2.imshow('CAM_LH-SWEEP', frame)
#
#     # Nhấn 'q' để thoát
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # Giải phóng camera và đóng các cửa sổ
# cap.release()
# cv2.destroyAllWindows()
#========================================-Su dung màu HSV-============================================================
# import cv2
# import numpy as np
#
# # Mở camera
# cap = cv2.VideoCapture(1)  # Sử dụng camera
#
# while True:
#     # Đọc khung hình từ camera
#     ret, frame = cap.read()
#     if not ret:
#         print("Không thể truy cập camera.")
#         break
#
#     # Resize khung hình cho dễ xử lý (tuỳ chọn)
#     frame = cv2.resize(frame, (800, 800))
#
#     # Chuyển đổi sang không gian màu HSV để dễ lọc màu trắng
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#
#     # Lọc màu trắng dựa trên HSV
#     lower_white = np.array([0, 0, 150])  # Giới hạn dưới của màu trắng
#     upper_white = np.array([180, 50, 255])  # Giới hạn trên của màu trắng
#
#     # Tạo mặt nạ chỉ giữ lại vùng màu trắng
#     mask = cv2.inRange(hsv, lower_white, upper_white)
#
#     # Làm mờ mặt nạ để giảm nhiễu (tăng kích thước kernel lên 7x7 để làm mịn tốt hơn)
#     blurred = cv2.GaussianBlur(mask, (11, 11), 0) #có thể để thông số (11 , 11) or (7,7)
#
#     # Áp dụng phát hiện cạnh Canny (thay đổi ngưỡng để phát hiện cạnh chính xác hơn)
#     edges = cv2.Canny(blurred, 50, 100)
#
#     # Tìm các đường viền (contours) trên ảnh nhị phân sau khi phát hiện cạnh
#     contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
#
#     # Duyệt qua các đường viền để kiểm tra xem có phải là tờ giấy trắng hay không
#     for contour in contours:
#         # Sử dụng xấp xỉ đa giác để làm mịn các cạnh
#         epsilon = 0.015 * cv2.arcLength(contour, True)  # Đặt epsilon là 15% của chu vi đường viền
#         approx = cv2.approxPolyDP(contour, epsilon, True)
#
#
#         # Xác định hình chữ nhật bao quanh đường viền xấp xỉ
#         x, y, w, h = cv2.boundingRect(approx)
#
#         # Kiểm tra tỷ lệ giữa chiều rộng và chiều cao để tìm các đối tượng gần hình chữ nhật (tờ giấy)
#         aspect_ratio = float(w) / h
#         if 0.8 < aspect_ratio < 1.2 and w > 100:  # Giới hạn tỷ lệ và kích thước
#             # Vẽ đường viền màu xanh lá lên tờ giấy
#             cv2.drawContours(frame, [approx], 0, (0, 255, 0), 3)
#
#     # Hiển thị khung hình với đường viền được vẽ
#     cv2.imshow('CAM_LH-SWEEP', frame)
#
#     # Nhấn 'q' để thoát
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# Giải phóng camera và đóng các cửa sổ
# cap.release()
# cv2.destroyAllWindows()
#==============================Su dung màu xanh và trắng nhận bảng rổ========================================================
# import cv2
# import numpy as np
#
# # Mở camera
# cap = cv2.VideoCapture(1)  # Sử dụng camera
#
# while True:
#     # Đọc khung hình từ camera
#     ret, frame = cap.read()
#     if not ret:
#         print("Không thể truy cập camera.")
#         break
#
#     # Resize khung hình cho dễ xử lý (tùy chọn)
#     frame = cv2.resize(frame, (800, 800))
#
#     # Chuyển đổi sang không gian màu HSV để dễ lọc màu
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#
#     # Lọc màu trắng dựa trên HSV
#     lower_white = np.array([0, 0, 180])  # Giới hạn dưới của màu trắng
#     upper_white = np.array([180, 50, 255])  # Giới hạn trên của màu trắng
#     white_mask = cv2.inRange(hsv, lower_white, upper_white)
#
#     # Lọc màu đen dựa trên HSV
#     lower_black = np.array([0, 0, 0])  # Giới hạn dưới của màu đen
#     upper_black = np.array([180, 255, 50])  # Giới hạn trên của màu đen
#     black_mask = cv2.inRange(hsv, lower_black, upper_black)
#
#     # Làm mờ mặt nạ để giảm nhiễu (tăng kích thước kernel để mượt hơn)
#     blurred_white = cv2.GaussianBlur(white_mask, (7, 7), 0)
#     blurred_black = cv2.GaussianBlur(black_mask, (7, 7), 0)
#
#     # Áp dụng phát hiện cạnh Canny (điều chỉnh ngưỡng để phát hiện cạnh tốt hơn)
#     edges_white = cv2.Canny(blurred_white, 30, 100)
#     edges_black = cv2.Canny(blurred_black, 30, 100)
#
#     # Tìm các đường viền cho màu trắng
#     contours_white, _ = cv2.findContours(edges_white, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
#     # Tìm các đường viền cho màu đen
#     contours_black, _ = cv2.findContours(edges_black, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
#     # Vẽ các đường viền màu xanh cho vùng trắng (xấp xỉ đa giác để làm mịn các cạnh)
#     for contour in contours_white:
#         if cv2.contourArea(contour) < 1000:  # Loại bỏ các đường viền quá nhỏ để tránh nhiễu
#             continue
#         # Xấp xỉ đa giác để làm mịn các cạnh
#         epsilon = 0.015 * cv2.arcLength(contour, True)
#         approx = cv2.approxPolyDP(contour, epsilon, True)
#         cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)  # Xanh lá cho vùng trắng
#
#     # Vẽ các đường viền màu đỏ cho vùng đen (xấp xỉ đa giác để làm mịn các cạnh)
#     for contour in contours_black:
#         if cv2.contourArea(contour) < 1000:  # Loại bỏ các đường viền quá nhỏ để tránh nhiễu
#             continue
#         # Xấp xỉ đa giác để làm mịn các cạnh
#         epsilon = 0.015 * cv2.arcLength(contour, True)
#         approx = cv2.approxPolyDP(contour, epsilon, True)
#         cv2.drawContours(frame, [approx], -1, (0, 0, 255), 3)  # Đỏ cho vùng đen
#
#
#     # Hiển thị khung hình với đường viền được vẽ
#     cv2.imshow('CAM_LH-SWEEP', frame)
#
#     # Nhấn 'q' để thoát
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # Giải phóng camera và đóng các cửa sổ
# cap.release()
# cv2.destroyAllWindows()
#====================================Another way cam xoay==============================================
# import cv2
# import numpy as np
#
# # Mở camera
# cap = cv2.VideoCapture(1)
#
# while True:
#     # Đọc khung hình từ camera
#     ret, frame = cap.read()
#     if not ret:
#         print("Không thể truy cập camera.")
#         break
#
#     # Resize khung hình cho dễ xử lý
#     frame = cv2.resize(frame, (800, 800))
#
#     # Chuyển đổi sang không gian màu HSV để dễ lọc màu
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#
#     # Lọc màu trắng dựa trên HSV
#     lower_white = np.array([0, 0, 180])  # Giới hạn dưới của màu trắng
#     upper_white = np.array([180, 50, 255])  # Giới hạn trên của màu trắng
#     white_mask = cv2.inRange(hsv, lower_white, upper_white)
#
#     # Tìm các đường viền của vùng trắng
#     contours_white, _ = cv2.findContours(white_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
#     # Xác định trung tâm khung hình
#     frame_center_x = frame.shape[1] // 2  # Trung tâm của chiều ngang (trục X)
#
#     # Chỉ tiếp tục nếu tìm thấy vùng màu trắng
#     if contours_white:
#         # Chọn đường viền lớn nhất
#         max_contour = max(contours_white, key=cv2.contourArea)
#
#         # Tính toán khung bao quanh đường viền lớn nhất
#         rect = cv2.minAreaRect(max_contour)
#
#         # Lấy tọa độ trung tâm của vùng màu trắng
#         white_center_x = int(rect[0][0])  # Tọa độ X của trung tâm vùng trắng
#
#         # Tính toán độ lệch của vùng trắng so với trung tâm khung hình
#         offset_x = white_center_x - frame_center_x
#
#         # Tính toán góc xoay (giới hạn góc xoay để tránh xoay quá lớn)
#         max_rotation_angle = 30  # Giới hạn góc xoay ngang tối đa (30 độ)
#         angle = max_rotation_angle * (offset_x / frame_center_x)  # Tính tỷ lệ xoay
#
#         # Chỉ xoay theo chiều ngang (quanh trục Y)
#         rotation_matrix = cv2.getRotationMatrix2D((frame_center_x, frame.shape[0] // 2), angle, 1.0)
#         rotated_frame = cv2.warpAffine(frame, rotation_matrix, (frame.shape[1], frame.shape[0]))
#
#         # Vẽ đường viền cho vùng trắng (sau khi xoay)
#         box = cv2.boxPoints(rect)
#         box = np.int32(box)
#         cv2.drawContours(rotated_frame, [box], 0, (0, 255, 0), 3)
#
#         # Hiển thị khung hình đã xoay
#         cv2.imshow('LH-Sweep', rotated_frame)
#     else:
#         # Hiển thị khung hình gốc nếu không có màu trắng
#         cv2.imshow('Cam goc', frame)
#
#     # Nhấn 'q' để thoát
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # Giải phóng camera và đóng các cửa sổ
# cap.release()
# cv2.destroyAllWindows()
#=================================code mới==============================================
# import cv2
# import numpy as np
#
# # Mở camera
# cap = cv2.VideoCapture(1)
#
# # Kích thước vùng ROI
# roi_x = 200
# roi_y = 200
# roi_w = 400
# roi_h = 400
#
# while True:
#     # Đọc khung hình từ camera
#     ret, frame = cap.read()
#     if not ret:
#         print("Không thể truy cập camera.")
#         break
#
#     # Giảm độ phân giải khung hình
#     frame = cv2.resize(frame, (640, 480))
#
#     # Lấy vùng ROI
#     roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
#
#     # Chuyển đổi sang không gian màu YCrCb
#     ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
#
#     # Tách kênh Y (luminance) để phát hiện vùng sáng (màu trắng)
#     y_channel = ycrcb[:, :, 0]
#
#     # Lọc vùng sáng (màu trắng) dựa trên giá trị kênh Y
#     _, mask = cv2.threshold(y_channel, 200, 255, cv2.THRESH_BINARY)
#
#     # Sử dụng bộ lọc Sobel thay vì Canny để phát hiện cạnh nhanh hơn
#     grad_x = cv2.Sobel(mask, cv2.CV_64F, 1, 0, ksize=3)
#     grad_y = cv2.Sobel(mask, cv2.CV_64F, 0, 1, ksize=3)
#     sobel_edges = cv2.convertScaleAbs(cv2.sqrt(grad_x**2 + grad_y**2))
#
#     # Tìm các đường viền (contours) trên ảnh nhị phân sau khi phát hiện cạnh
#     contours, _ = cv2.findContours(sobel_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
#     # Biến để lưu trữ hình vuông mới phát hiện
#     current_square = None
#
#     # Duyệt qua các đường viền để kiểm tra
#     for contour in contours:
#         if cv2.contourArea(contour) < 1000:  # Bỏ qua đường viền nhỏ
#             continue
#
#         # Sử dụng xấp xỉ đa giác để làm mịn các cạnh
#         epsilon = 0.03 * cv2.arcLength(contour, True)
#         approx = cv2.approxPolyDP(contour, epsilon, True)
#
#         # Xác định hình chữ nhật bao quanh đối tượng
#         x, y, w, h = cv2.boundingRect(approx)
#
#         # Điều chỉnh để tạo ra hình vuông bao quanh đối tượng
#         side_length = max(w, h)
#         top_left = (x + roi_x, y + roi_y)  # Điều chỉnh lại theo vùng ROI
#         bottom_right = (x + side_length + roi_x, y + side_length + roi_y)
#
#         current_square = (top_left, bottom_right)
#         break  # Chỉ lấy đường viền lớn nhất
#
#     # Vẽ hình vuông màu xanh nếu phát hiện
#     if current_square:
#         top_left, bottom_right = current_square
#         cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 3)
#
#     # Hiển thị khung hình với hình vuông được vẽ
#     cv2.imshow('Optimized Frame with ROI', frame)
#
#     # Nhấn 'q' để thoát
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # Giải phóng camera và đóng các cửa sổ
# cap.release()
# cv2.destroyAllWindows()
#==================================================================================

import cv2
import numpy as np
import imutils
from threading import Thread

# Hàm đọc video từ camera sử dụng đa luồng
class VideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

# Hàm phát hiện vùng giấy trắng
def detect_paper(frame):
    # Chuyển sang không gian màu xám
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Lọc vùng sáng (màu trắng)
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Sử dụng bộ lọc Sobel để phát hiện cạnh
    grad_x = cv2.Sobel(mask, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(mask, cv2.CV_64F, 0, 1, ksize=3)
    sobel_edges = cv2.convertScaleAbs(cv2.sqrt(grad_x**2 + grad_y**2))

    # Tìm các đường viền
    contours, _ = cv2.findContours(sobel_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    current_square = None
    largest_area = 0  # Để tìm đối tượng lớn nhất

    # Duyệt qua các đường viền để kiểm tra
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 2000:  # Bỏ qua đường viền nhỏ
            continue

        # Sử dụng xấp xỉ đa giác
        epsilon = 0.03 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Xác định hình chữ nhật bao quanh đối tượng
        x, y, w, h = cv2.boundingRect(approx)

        # Kiểm tra tỷ lệ chiều rộng/chiều cao
        aspect_ratio = float(w) / h
        if 0.9 < aspect_ratio < 1.1 and area > largest_area:  # Kiểm tra nếu tỷ lệ gần 1:1 và là đối tượng lớn nhất
            side_length = max(w, h)
            top_left = (x, y)
            bottom_right = (x + side_length, y + side_length)
            current_square = (top_left, bottom_right)
            largest_area = area

    if current_square:
        # Vẽ hình vuông màu xanh quanh đối tượng
        top_left, bottom_right = current_square
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 3)

    return frame

# Khởi động video stream sử dụng đa luồng
video_stream = VideoStream(src=1).start()

while True:
    frame = video_stream.read()

    # Resize khung hình
    frame = imutils.resize(frame, width=640)

    # Phát hiện tờ giấy màu trắng
    processed_frame = detect_paper(frame)

    # Hiển thị khung hình đã được xử lý
    cv2.imshow('LH-SWEEP', processed_frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
video_stream.stop()
cv2.destroyAllWindows()

