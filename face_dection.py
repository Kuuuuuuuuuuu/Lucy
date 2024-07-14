# 실제 이미지 출력
image = cv2.imread('C:/Users/konyang/Desktop/Ku/kuu/lucy/testface.jpg', cv2.IMREAD_COLOR)
image.shape
cv2.imshow('title', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 이미지 확인 함수
def plt_imshow(title='image', img=None, figsize=(8 ,5)):
    plt.figure(figsize=figsize)
 
    if type(img) == list:
        if type(title) == list:
            titles = title
        else:
            titles = []
 
            for i in range(len(img)):
                titles.append(title)
 
        for i in range(len(img)):
            if len(img[i].shape) <= 2:
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_GRAY2RGB)
            else:
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_BGR2RGB)
 
            plt.subplot(1, len(img), i + 1), plt.imshow(rgbImg)
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
 
        plt.show()
    else:
        if len(img.shape) < 3:
            rgbImg = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 
        plt.imshow(rgbImg)
        plt.title(title)
        plt.xticks([]), plt.yticks([])
        plt.show()


# 얼굴 인식 선언
detector = dlib.get_frontal_face_detector()
predictor = face_recognition.api.pose_predictor_68_point


image_path = 'C:/Users/konyang/Desktop/Ku/kuu/lucy/testface.jpg' 
org_image = cv2.imread(image_path) 
image = org_image.copy() 
image = imutils.resize(image, width=500) 
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
 
rects = detector(gray, 1)


for (i, rect) in enumerate(rects):
    # 얼굴 영역의 얼굴 랜드마크를 결정한 다음 
    # 얼굴 랜드마크(x, y) 좌표를 NumPy Array로 변환.
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
    
    # dlib의 사각형을 OpenCV bounding box로 변환(x, y, w, h)
    (x, y, w, h) = face_utils.rect_to_bb(rect)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
   
plt_imshow("Output", image, figsize=(16,10))


roi = image[y:y+h, x:x+w]     # roi 지정
image2 = roi.copy()           # roi 배열 복제했기 때문에 초록색 사각형이 표시되지 않았다.

image[y:y+h, x+w:x+w+w] = roi # 새로운 좌표에 roi 추가, 태양 2개 만들기
cv2.rectangle(image, (x,y), (x+w+w, y+h), (0,255,0)) # 2개의 태양 영역에 사각형 표시


imgs = {'img':image, 'img2':image2} #image2 = 얼굴만 검출
for i, (k, v) in enumerate(imgs.items()):
    plt.subplot(1,2, i + 1)
    plt.imshow(v[:,:,::-1])
    plt.title(k)
    plt.xticks([]); plt.yticks([])

plt.show()


# 얼굴 이미지 크기
roi = cv2.resize(roi, (48, 48)) # 크기 조절 (모델 전달 위함)
roi = roi / 255 # 배치 단위로 전달, 배치크기 = 1
roi = np.expand_dims(roi, axis=0) # 배치 크기 추가
roi.shape



pred_probability = network.predict(roi)
pred_probability


pred = np.argmax(pred_probability)
pred

test_dataset.class_indices