from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from data_loader import df
from Models.CNN_model import model

train_images, val_images, test_images = df()
CLASSES = 4

custom_model = model()

def eval_model():
    results = custom_model.evaluate(test_images)

    print(f'Pérdida (Loss): {results[0]}')
    print(f'Precisión (Accuracy): {results[1]}')
    print(f'Precisión (Precision): {results[2]}')
    print(f'Recall: {results[3]}')
    print(f'Specificidad (Specificity): {results[4]}')
    print(f'F1-Score: {results[5]}')


def eval_cat():
    test_scores = custom_model.evaluate(test_images)
    print("Testing Accuracy: %.2f%%"%(test_scores[1] * 100))
    pred_labels = custom_model.predict(test_images)

    def roundoff(arr):
        """To round off according to the argmax of each predicted label array."""
        arr[np.argwhere(arr != arr.max())] = 0
        arr[np.argwhere(arr == arr.max())] = 1
        return arr

    for labels in pred_labels:
        labels = roundoff(labels)
        
    pred = np.argmax(pred_labels,axis=1)
    test_ls, pred_ls = test_images.classes,pred
    conf_arr = confusion_matrix(test_ls, pred_ls)

    plt.figure(figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

    ax = sns.heatmap(conf_arr, cmap='Greens', annot=True, fmt='d', xticklabels=CLASSES, yticklabels=CLASSES)

    plt.title('Alzheimer\'s Disease Diagnosis')
    plt.xlabel('Prediction')
    plt.ylabel('Truth')
    plt.show(ax)
