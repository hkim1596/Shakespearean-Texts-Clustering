import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np

# Load the saved model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("./saved_model")
model = DistilBertForSequenceClassification.from_pretrained("./saved_model")

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Generate predictions for the test set
def get_predictions(test_encodings):
    with torch.no_grad():
        inputs = test_encodings["input_ids"].to("cuda" if torch.cuda.is_available() else "cpu")
        masks = test_encodings["attention_mask"].to("cuda" if torch.cuda.is_available() else "cpu")
        outputs = model(inputs, attention_mask=masks)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)
    return predictions.cpu().numpy()

# Example for the Base Classifier (Good vs Bad)
good_preds = get_predictions(good_test)
bad_preds = get_predictions(bad_test)

all_true_labels = [0] * len(good_test["input_ids"]) + [1] * len(bad_test["input_ids"])
all_predictions = list(good_preds) + list(bad_preds)

# Compute confusion matrix
cnf_matrix = confusion_matrix(all_true_labels, all_predictions)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Good', 'Bad'], title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Good', 'Bad'], normalize=True, title='Normalized confusion matrix')

plt.show()
