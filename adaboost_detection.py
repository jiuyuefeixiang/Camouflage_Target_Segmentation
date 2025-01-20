import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import precision_recall_curve, average_precision_score
from skimage.feature import hog
def extract_hog_features(image, winSize=(64, 128)):
    # Resize image to a fixed size
    image_resized = cv2.resize(image, winSize)
    # Convert to grayscale
    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    # Extract HOG features
    hog_features, _ = hog(gray, pixels_per_cell=(8, 8),
                          cells_per_block=(2, 2), visualize=False, multichannel=False)
    return hog_features
def load_data(image_paths, annotations):
    X = []
    y = []
    for img_path, annotation in zip(image_paths, annotations):
        img = cv2.imread(img_path)
        if annotation['label'] == 'camouflage':  # 只考虑伪装目标
            bbox = annotation['bbox']
            x, y, w, h = bbox
            patch = img[y:y+h, x:x+w]
            hog_feature = extract_hog_features(patch)
            X.append(hog_feature)
            y.append(1)  # 正样本
        else:
            # 这里可以随机选取一些负样本（背景区域）
            # 为简化起见，我们省略负样本的具体提取过程
            pass
    X = np.array(X)
    y = np.array(y)
    return X, y
def detect_camouflage(image, ada_boost, winSize=(64, 128), stepSize=(8, 8)):
    (h, w) = image.shape[:2]
    detections = []
    for y in range(0, h - winSize[1], stepSize[1]):
        for x in range(0, w - winSize[0], stepSize[0]):
            patch = image[y:y + winSize[1], x:x + winSize[0]]
            if patch.shape[:2] != winSize:
                continue
            hog_feature = extract_hog_features(patch)
            hog_feature = hog_feature.reshape(1, -1)
            prediction = ada_boost.predict(hog_feature)
            if prediction == 1:
                detections.append((x, y, x + winSize[0], y + winSize[1]))
    return detections
image_paths = "/home/iair/Downloads/train/image"  # 图片路径列表
annotations = "/home/iair/Downloads/train/label" # 标注信息列表

# 加载数据
X, y = load_data(image_paths, annotations)

# 创建基学习器（决策树）
base_estimator = DecisionTreeClassifier(max_depth=1)

# 创建AdaBoost分类器
ada_boost = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=50, learning_rate=1.0, random_state=42)

# 训练模型
ada_boost.fit(X, y)
test_image_paths = [...]

# 评估模型
all_gt_bboxes = []  # 存储所有图片的GT bbox
all_pred_scores = []  # 存储所有图片的预测得分（这里简化为1或0）
all_pred_bboxes = []  # 存储所有图片的预测bbox

for img_path in test_image_paths:
    img = cv2.imread(img_path)
    gt_bboxes = get_gt_bboxes_for_image(img_path)  # 获取该图片的GT bbox，假设有这个函数
    detections = detect_camouflage(img, ada_boost)

    pred_scores = [1] * len(detections)  # 简化为1，实际可以计算置信度
    pred_bboxes = detections

    all_gt_bboxes.append(gt_bboxes)
    all_pred_scores.extend(pred_scores)
    all_pred_bboxes.extend(pred_bboxes)

# 将GT bbox和预测bbox转换为适合评估的格式
# 这里假设gt_bboxes和pred_bboxes已经处理好

# 计算PR曲线和AP
precision, recall, _ = precision_recall_curve(all_gt_labels, all_pred_scores, pos_label=1)
average_precision = average_precision_score(all_gt_labels, all_pred_scores)

# 绘制PR曲线
plt.plot(recall, precision, marker='.')
plt.title('Precision-Recall Curve: AP={0:0.2f}'.format(average_precision))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()