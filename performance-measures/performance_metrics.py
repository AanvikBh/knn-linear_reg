import numpy as np 

class PerformanceMetrics:
    def __init__(self, y_true, y_pred):
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.classes = np.unique(y_true)

    def accuracy(self):
        return np.mean(self.y_true == self.y_pred)

    def precision(self, key):
        if key == 'macro':
            precisions = []
            for cls in self.classes:
                true_positive = np.sum((self.y_pred == cls) & (self.y_true == cls))
                predicted_positive = np.sum(self.y_pred == cls)
                precision = true_positive / predicted_positive if predicted_positive > 0 else 0
                precisions.append(precision)
            return np.mean(precisions)
        elif key == 'micro':
            true_positive = np.sum(self.y_pred == self.y_true)
            predicted_positive = len(self.y_pred)
            return true_positive / predicted_positive if predicted_positive > 0 else 0

    def recall(self, key):
        if key == 'macro':
            recalls = []
            for cls in self.classes:
                true_positive = np.sum((self.y_pred == cls) & (self.y_true == cls))
                actual_positive = np.sum(self.y_true == cls)
                recall = true_positive / actual_positive if actual_positive > 0 else 0
                recalls.append(recall)
            return np.mean(recalls)
        elif key == 'micro':
            true_positive = np.sum(self.y_pred == self.y_true)
            actual_positive = len(self.y_true)
            return true_positive / actual_positive if actual_positive > 0 else 0

    def f1_score(self, key):
        precision = self.precision(key)
        recall = self.recall(key)
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    def evaluate(self):
        return {
            "accuracy": self.accuracy(),
            "precision_macro": self.precision(key='macro'),
            "recall_macro": self.recall(key='macro'),
            "f1_score_macro": self.f1_score(key='macro'),
            "precision_micro": self.precision(key='micro'),
            "recall_micro": self.recall(key='micro'),
            "f1_score_micro": self.f1_score(key='micro')
        }
