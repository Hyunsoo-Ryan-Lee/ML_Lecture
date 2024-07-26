import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    classification_report
)

def evaluate_cls_model(y_, y_pred, if_binary: bool=False):

    # 정확도
    acc = accuracy_score(y_, y_pred)
    print(f"Accuracy: {acc:.3f}")
    print("-"*50)

    if if_binary:
        # 정밀도
        precision = precision_score(y_, y_pred)
        print(f"Precision: {precision:.3f}")
        print("-"*50)

        # 재현율
        recall = recall_score(y_, y_pred)
        print(f"Recall: {recall:.3f}")
        print("-"*50)

        # F1 점수
        f1 = f1_score(y_, y_pred)
        print(f"F1 Score: {f1:.3f}")
        print("-"*50)

        # ROC-AUC
        auc = roc_auc_score(y_, y_pred)
        print(f"AUC: {auc:.2f}")
        print("-"*50)
    
    # 상세 분류 리포트
    report = classification_report(y_, y_pred)
    print("Classification Report:")
    print(report)

    # 혼동 행렬
    cm = confusion_matrix(y_, y_pred)
    print("Confusion Matrix:")
    print(cm)
    print("-"*50)


    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predict')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()