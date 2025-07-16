import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO('./train3/weights/best.pt')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, conf=0.6, imgsz=640)
    
    for result in results:
        if hasattr(result.obb, 'xywhr'):
            boxes = result.obb.xywhr.cpu().numpy()
            for box in boxes:
                x, y, w, h, angle = box
                angle_deg = np.degrees(angle) if angle < np.pi else angle
                
                # 动态调整宽高（实验性）
                #if angle_deg % 90 > 45:
                #    w, h = h, w
                
                # 生成旋转矩形并绘制
                rect = ((x, y), (w, h), angle_deg)
                box_points = cv2.boxPoints(rect).astype(np.int32)
                cv2.polylines(frame, [box_points], True, (0, 255, 0), 2)
                
                # 标注类别和置信度
                cls_id = int(result.obb.cls[0])
                conf = float(result.obb.conf[0])
                label = f"{model.names[cls_id]} {conf:.2f}"
                cv2.putText(frame, label, (int(x), int(y) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.imshow('YOLO OBB Detection', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
'''
import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO('./train3/weights/best.pt')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, conf=0.5, imgsz=640)  # 降低阈值 + 调整分辨率
    
    for result in results:
        if hasattr(result.obb, 'xyxyxyxy'):  # 使用实际支持的属性
            boxes = result.obb.xyxyxyxy.cpu().numpy()
            for box in boxes:
                box = np.array(box, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [box], True, (0, 255, 0), 2)
                # 标注类别和置信度
                cls_id = int(result.obb.cls[0])
                conf = float(result.obb.conf[0])
                label = f"{model.names[cls_id]} {conf:.2f}"
                cv2.putText(frame, label, (box[0][0][0], box[0][0][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.imshow('YOLO OBB Detection', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

'''
'''
import cv2

# 打开摄像头（默认摄像头通常是 0，如果有多个摄像头可以尝试 1, 2 等）
cap = cv2.VideoCapture(0)

while True:
    # 读取摄像头帧
    ret, frame = cap.read()
    if not ret:
        print("无法读取摄像头画面！")
        break



    # 显示结果
    cv2.imshow('YOLO 旋转框检测', frame)

    # 按 'q' 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头并关闭窗口
cap.release()
cv2.destroyAllWindows()
'''