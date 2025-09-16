import numpy as np

def prep_bgr_to_nchw(img_bgr, W, H):
    """Prépare une image BGR en NCHW float32 [1,3,H,W] normalisée."""
    import cv2
    r = cv2.resize(img_bgr, (W,H))
    r = cv2.cvtColor(r, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    r = np.transpose(r, (2,0,1)).reshape(1,3,H,W).copy()
    return r

def decode_yolov8(outputs, img_w, img_h, conf_th=0.25, keep=(0,7)):
    """Décodage YOLOv8 [N,85] → boîtes [x1,y1,x2,y2,score,cls] avec NMS 0.5."""
    pred = outputs[0].reshape(-1, 85)
    pred = pred[pred[:,4] > conf_th]
    if len(pred)==0: 
        return np.empty((0,6))
    cls = np.argmax(pred[:,5:], axis=1)
    scr = pred[:,4] * pred[np.arange(len(pred)), 5+cls]
    sel = np.isin(cls, keep)
    pred, cls, scr = pred[sel], cls[sel], scr[sel]
    x,y,w,h = pred[:,0], pred[:,1], pred[:,2], pred[:,3]
    x1,y1,x2,y2 = x-w/2, y-h/2, x+w/2, y+h/2
    boxes = np.stack([x1,y1,x2,y2,scr,cls], axis=1)
    boxes[:,[0,2]] = np.clip(boxes[:,[0,2]], 0, img_w-1)
    boxes[:,[1,3]] = np.clip(boxes[:,[1,3]], 0, img_h-1)
    return nms(boxes, 0.5)

def nms(boxes, iou_th=0.5):
    """NMS standard sur [x1,y1,x2,y2,score,cls]."""
    if len(boxes)==0: return boxes
    b = boxes[np.argsort(-boxes[:,4])]
    keep=[]
    while len(b):
        cur, rest = b[0], b[1:]
        keep.append(cur)
        if len(rest)==0: break
        xx1 = np.maximum(cur[0], rest[:,0])
        yy1 = np.maximum(cur[1], rest[:,1])
        xx2 = np.minimum(cur[2], rest[:,2])
        yy2 = np.minimum(cur[3], rest[:,3])
        iw = np.maximum(0, xx2-xx1)
        ih = np.maximum(0, yy2-yy1)
        inter = iw*ih
        iou = inter/(((cur[2]-cur[0])*(cur[3]-cur[1]))+((rest[:,2]-rest[:,0])*(rest[:,3]-rest[:,1]))-inter+1e-9)
        b = rest[iou < iou_th]
    return np.array(keep)

