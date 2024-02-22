from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier


def perform_centroid_test(X_train, X_test,y_train,y_test):
    centroid_classifier = NearestCentroid()
    centroid_model = centroid_classifier.fit(X_train, y_train)
    y_pred = centroid_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"nearest centroid accuracy = {accuracy}")
    return y_pred, centroid_model.classes_


def perform_knn_test(X_train, X_test,y_train,y_test):
    knn_classifier = KNeighborsClassifier(algorithm='kd_tree', leaf_size=200, n_neighbors=25,weights='distance')
    knn_model = knn_classifier.fit(X_train, y_train)
    y_pred = knn_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"knn accuracy = {accuracy}")
    return y_pred, knn_model.classes_


def perform_svm_test(X_train, X_test,y_train,y_test):
    svm_classifier = svm.SVC(C = 5,kernel = 'rbf',gamma='scale',probability=True)
    svm_model = svm_classifier.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"svm accuracy = {accuracy}")
    return y_pred, svm_model.classes_




