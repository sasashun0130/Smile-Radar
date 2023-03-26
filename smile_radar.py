# coding:utf-8

import numpy
import dlib
from imutils import face_utils
import cv2

#合成させる画像を指定
img_file = cv2.imread('koume.png', -1)

# --------------------------------
# 1.顔ランドマーク検出の前準備
# --------------------------------
# 顔ランドマーク検出ツールの呼び出し
face_detector = dlib.get_frontal_face_detector()
predictor_path = 'shape_predictor_68_face_landmarks.dat'
face_predictor = dlib.shape_predictor(predictor_path)


# --------------------------------
# 2.認識した顔を口の動きにより笑顔になっているか判定する関数
# --------------------------------
def smile_find(img_gry,faces):

  # 検出した全顔に対して処理
  for face in faces:
    # 顔のランドマーク検出
    landmark = face_predictor(img_gry, face)
    # 処理高速化のためランドマーク群をNumPy配列に変換(必須)
    landmark = face_utils.shape_to_np(landmark)
    
    #口の部分にて笑顔判定
    #ランドマークの座標で判定する用いる番号は49,55,67
    x_49 = landmark[48][1]
    y_49 = landmark[48][0]
    
    x_55 = landmark[54][1]
    y_55 = landmark[54][0]
    
    x_67 = landmark[66][1]
    y_67 = landmark[66][0]

    u = numpy.array([x_55 - x_49, y_55 - y_49])
    v = numpy.array([x_67 - x_49, y_67 - y_49])
    
    L = abs(numpy.cross(u, v) / numpy.linalg.norm(u))
    
    print(L)

    if L>15:  #笑顔判定
      return True
    
    return False      


# --------------------------------
# 3.カメラ画像の取得
# --------------------------------
# カメラの指定
cap = cv2.VideoCapture(0)

# カメラ画像の表示 ('q'入力で終了)
while(True):
  ret, img = cap.read()
  
  # 顔検出
  img_gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  faces = face_detector(img_gry, 1)
  
  #学習済みデータの分類器カスケードファイルを読み込み
  face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
  #顔認識画像の高さ・横の長さを取得
  h, w = img.shape[:2] 
  
  # 顔のランドマーク検出
  smile = smile_find(img_gry,faces)
  
  if smile is True:
    img_GRAY = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.COLOR_BGR2GRAYを変えることで色々オプションができる
    faces = face_cascade.detectMultiScale(img_GRAY, 1.1, 5)
    #1.3倍でフレームを大きくしてスキャンしていく。1.1倍なら精度up
    
    #透過画像のエラー解消のためaddWeighted関数と同じ機能
    for (x,y,w,h) in faces:
      img1  = img[y:y+h,x:x+w,:]
      img2  = cv2.resize(img_file, (w,h))
      #blended = cv2.addWeighted(src1=img1,alpha=1,src2=img2,beta=5,gamma=0)#基礎値img1=1,img2=5  
                
      mask = img2[:,:,3]  # アルファチャンネルだけ抜き出す。
      mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # 3色分に増やす。
      mask = mask / 255  # 0-255だと使い勝手が悪いので、0.0-1.0に変更。          

      img1=img1.astype('float')
      img2=img2.astype('float')
      img1 *= 1 - mask  # 透過率に応じて元の画像を暗くする。
      img1 += img2[:,:,:3] * mask  # 貼り付ける方の画像に透過率をかけて加算。
      img1=img1.astype('uint8')
      img[y:y+h,x:x+w,:] = img1 

  # 結果の表示
  cv2.imshow('img', img)

  # 'q'が入力されるまでループ
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

# 後処理
cap.release()
cv2.destroyAllWindows()