import os
import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc,confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 


def PCA(image_path, num_components=20):
    # Convert image to grayscale
    image = cv2.imread(image_path, 0)
    image = cv2.resize(image, (50, 50))

    # Convert image to numpy array
    X = np.array(image)

    # Reshape 2D array into a 1D array of pixels
    X = X.reshape(-1, X.shape[1])

    # Step-1: Mean centering
    X_meaned = X - np.mean(X, axis=0)

    # Step-2: Calculate covariance matrix
    cov_mat = np.cov(X_meaned, rowvar=False)

    # Step-3: Compute eigenvalues and eigenvectors
    eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)

    # Step-4: Sort eigenvalues and corresponding eigenvectors in descending order
    sorted_index = np.argsort(eigen_values)[::-1]
    # sorted_eigenvalues = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:, sorted_index]

    # Step-5: Select subset of eigenvectors
    eigenvector_subset = sorted_eigenvectors[:, 0:num_components]

    # Step-6: Reduce dimensionality of the data
    X_reduced = np.dot(eigenvector_subset.transpose(), X_meaned.transpose()).transpose()
    


    return X_reduced  # Flatten the array before returning



def train_svm_with_pca(num_components=20, test_size=0.2, random_state=42):
    # Call the pca_face_recognition function to obtain the predict_label function
   

    # Prepare the data for training the SVM model
    features = []
    labels = []

    for person_id in range(1, 6):
        person_dir = f"our_faces/data/{person_id}"
        for filename in os.listdir(person_dir):
            
            image_path = os.path.join(person_dir, filename) 
            predicted_label = PCA(image_path, num_components)
            features.append(predicted_label)
            labels.append(person_id)



    features = np.array(features)


    features = features.reshape(150, -1)


    # Reshape the second array to match the shape of the first array
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=random_state)

    # Create and train the SVM classifier
    clf = svm.SVC(kernel='linear', probability=True)
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_test)

    # Evaluate the performance of the SVM model
    accuracy = accuracy_score(y_test, y_pred)
    print("accuracy",accuracy)
    target_names = ['1','2','3','4','5']
    print(classification_report(y_test, y_pred, target_names=target_names))
    return clf




def predict_with_svm(clf, image_path):

    predicted_label = PCA(image_path, 20)

    predicted_label = predicted_label.reshape(1,-1)
    prediction = clf.predict(predicted_label)
    probs = clf.predict_proba(predicted_label)
    found = max(max(probs))
    image = "our_faces/data/"+ str(prediction[0]) +"/"+ str(1) + "_" + str(prediction[0]) + ".jpg"
    print("prediction",prediction)
    print("probability",probs)
    print("is found?",found)
    return image,prediction,found


def get_features(test_dir="our_faces/test"):
    features = []
    labels = []
    for filename in os.listdir(test_dir):
        image_path = os.path.join(test_dir, filename)
        person_number = int(filename.split('_')[1].split('.')[0])
        predicted_label = PCA(image_path)
        features.append(predicted_label)
        labels.append(person_number)

    return features, labels


def calculate_and_plot_roc_curve(classifier):

    features, labels = get_features()
    features = np.array(features)
    features = features.reshape(30, -1)
    X_test = features
    y_test = labels

    # Calculate the decision scores for the test set
    decision_scores = classifier.decision_function(X_test)

    # Compute the false positive rate (FPR), true positive rate (TPR), and threshold values
    fpr, tpr, _ = roc_curve(y_test, decision_scores[:, 1], pos_label=1)
    roc_auc = auc(fpr, tpr)

    # Create the ROC curve plot
    fig, ax = plt.subplots(1, 1, figsize=(4, 3.5))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")

    return fig

def create_confusion_matrix(classifier):
    # Prepare the data for making predictions
    features, labels = get_features()
    features = np.array(features)
    features = features.reshape(30, -1)
    X_test = features
    y_test = labels

    # Calculate the decision scores for the test set
    decision_scores = classifier.decision_function(X_test)

    y_pred = np.argmax(decision_scores, axis=1)
    conf_mat = confusion_matrix(y_test,y_pred)

    cm_df = pd.DataFrame(conf_mat,
                     index = ['0','E','D','Y','M','B'], 
                     columns = ['E','D','Y','M','B','0'])
    fig, ax = plt.subplots()
    sns.heatmap(cm_df, annot=True)
    plt.title('Confusion Matrix')
    plt.ylabel('Actal Values')
    plt.xlabel('Predicted Values')
    return fig
