class MyBernoulliNBClassifier():
    def __init__(self, priors=None):
        self._Py = []  # Априорные вероятности y P(y)
        self._P_x_1_y = []  # Вероятности признаков P(x_i=1|y)
        self.classes = []  # Список уникальных классов
        self.count_features = None  # Общее количество признаков
        self.count_classes = None  # Общее количество классов
        self.count_elem_total = None  # Общее количество элементов

    def fit(self, X, y):
        self.classes = list(set(y))
        self.count_classes = len(self.classes)
        self.count_elem_total = len(y)
        self.count_features = len(X[0])

        self._Py = [0] * self.count_classes
        self._P_x_1_y = [[0] * self.count_features for _ in range(self.count_classes)]

        count_elem_in_classes = [0] * self.count_classes
        summ_obj_for_each_class = [[0] * self.count_features for _ in range(self.count_classes)]

        for obj_i, clas in enumerate(y):
            class_index = self.classes.index(clas)
            count_elem_in_classes[class_index] += 1
            for j in range(self.count_features):
                summ_obj_for_each_class[class_index][j] += X[obj_i][j]

        for i, count_elem in enumerate(count_elem_in_classes):  # Априорные вероятности классов P(y)
            self._Py[i] = count_elem / self.count_elem_total

        for y_i, obj in enumerate(summ_obj_for_each_class):  # Условные вероятности признаков P(x_i=1|y)
            for j in range(self.count_features):
                self._P_x_1_y[y_i][j] = obj[j] / count_elem_in_classes[y_i]

    def predict(self, X):
        predict_list = []
        for obj in X:
            products = [self._Py[i] for i in range(self.count_classes)]
            for class_index, _ in enumerate(self.classes):
                for feature_i, feature in enumerate(obj):
                    products[class_index] *= (
                            self._P_x_1_y[class_index][feature_i] * feature +
                            (1 - self._P_x_1_y[class_index][feature_i]) * (1 - feature)
                    )
            max_product = max(products)
            max_class = self.classes[products.index(max_product)]
            predict_list.append(max_class)
        return predict_list

    def predict_proba(self, X):
        predict_list = []
        for obj in X:
            products = [self._Py[i] for i in range(self.count_classes)]
            for class_index in range(self.count_classes):
                for feature_i, feature in enumerate(obj):
                    prob = self._P_x_1_y[class_index][feature_i]
                    products[class_index] *= prob * feature + (1 - prob) * (1 - feature)
            total = sum(products)
            if total == 0:
                normalized_probs = [1 / self.count_classes] * self.count_classes
            else:
                normalized_probs = [p / total for p in products]
            predict_list.append(normalized_probs)
        return predict_list

    @staticmethod
    def score(X, y):
        count_correct = sum(1 for i in range(len(y)) if y[i] == X[i])
        return count_correct / len(y)


def compare():
    from sklearn.preprocessing import Binarizer
    from sklearn.naive_bayes import BernoulliNB
    from sklearn.metrics import accuracy_score, mean_squared_error
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_digits

    digits = load_digits()
    X_digits, y_digits = digits.data, digits.target
    X_digits_bin = Binarizer(threshold=8).fit_transform(X_digits)  # Бинаризация данных


    X_train, X_test, y_train, y_test = train_test_split(X_digits_bin, y_digits, test_size=0.3, random_state=42)

    # Обучаем и предсказываем с помощью своего классификатора
    my_classifier = MyBernoulliNBClassifier()
    my_classifier.fit(X_train, y_train)
    y_pred_my = my_classifier.predict(X_test)
    y_pred_proba_my = my_classifier.predict_proba(X_test)

    # Обучаем и предсказываем с помощью BernoulliNB из sklearn
    sklearn_classifier = BernoulliNB()
    sklearn_classifier.fit(X_train, y_train)
    y_pred_sklearn = sklearn_classifier.predict(X_test)
    y_pred_proba_sklearn = sklearn_classifier.predict_proba(X_test)

    # Сравниваем точности двух классификаторов
    accuracy_my = my_classifier.score(y_test, y_pred_my)
    accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
    mse = mean_squared_error(y_pred_proba_sklearn, y_pred_proba_my)

    print("Среднеквадратическая ошибка (MSE) между scikit-learn и custom:", mse)
    print("Точность моего классификатора:", accuracy_my)
    print("Точность классификатора sklearn:", accuracy_sklearn)


compare()