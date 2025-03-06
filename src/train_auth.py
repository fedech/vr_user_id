from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.ensemble import BaggingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, log_loss, confusion_matrix
from tqdm.notebook import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, log_loss, confusion_matrix
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import VotingClassifier

from imblearn.over_sampling import SMOTE

class AuthenticationEvaluator:
    """
    A class to evaluate multiple classifiers using various evaluation metrics.

    Attributes:
        classifiers (list): A list containing the names and instances of classifiers to be evaluated.
        SEED (int): Random seed for reproducibility.
    """

    def __init__(self, SEED=42):
        self.classifiers = [
            ['ExtraTreesClassifier', ExtraTreesClassifier(bootstrap=False, max_depth=None,
                                                          min_samples_split=2, n_estimators=800,
                                                          random_state=SEED)],
            ['RandomForestClassifier', RandomForestClassifier(random_state=SEED)],
            ['LGBMClassifier', LGBMClassifier(random_state=SEED, verbosity=-1)]
        ]
        
        # Add the VotingClassifier
        voting_classifiers = [
            ('ET', self.classifiers[0][1]),  # ExtraTreesClassifier
            ('LGBM', self.classifiers[2][1]),  # LGBMClassifier
            ('RF', self.classifiers[1][1])  # RandomForestClassifier
        ]
        
        voting_classifier = VotingClassifier(
            estimators=voting_classifiers,
            voting='soft',
            n_jobs=-1
        )
        
        self.classifiers.append(['VotingClassifier', voting_classifier])
        self.SEED = SEED

    def accuracy_by_vote(self, model, X_test, y_test):
        """
        Calculate accuracy based on voting over predictions made during the two-minute test period.

        Args:
            model (object): Trained classifier model.
            X_test (DataFrame): Test data features.
            y_test (Series): True labels for test data.

        Returns:
            float: Accuracy score based on voting.
        """
        # Assuming y_test contains the ID information
        unique_ids = y_test.unique()
        
        # Dictionary to store votes for each ID
        votes = {id: {} for id in unique_ids}

        # Make predictions for all test data
        predictions = model.predict(X_test)

        # Count votes for each ID
        for true_id, pred_id in zip(y_test, predictions):
            votes[true_id][pred_id] = votes[true_id].get(pred_id, 0) + 1

        # Determine the final prediction for each ID based on the most common vote
        final_predictions = {id: max(id_votes, key=id_votes.get) for id, id_votes in votes.items()}

        # Calculate accuracy
        correct = sum(final_predictions[id] == id for id in unique_ids)
        total = len(unique_ids)
        accuracy = correct / total

        return accuracy
    
 

    def split_data(self, slow_data, fast_data, id_range=None, train_minutes=8, test_minutes=2):
        if id_range is None:
            X = slow_data.drop(columns=['ID'])
            y = slow_data['ID']
            fX = fast_data.drop(columns=['ID'])
            fy = fast_data['ID']
        else : 
            X = slow_data[slow_data["ID"].isin(id_range)].drop(columns=['ID'])
            y = slow_data[slow_data["ID"].isin(id_range)]['ID']
            fX = fast_data[fast_data["ID"].isin(id_range)].drop(columns=['ID'])
            fy = fast_data[fast_data["ID"].isin(id_range)]['ID']


        unique_minutes = slow_data['time_interval'].unique()

        # Split data based on selected minutes
        X_train = X[X['time_interval'].isin(unique_minutes[:train_minutes])]
        y_train = y[X['time_interval'].isin(unique_minutes[:train_minutes])]

        X_test = X[X['time_interval'].isin(unique_minutes[train_minutes:train_minutes+test_minutes])]
        y_test = y[X['time_interval'].isin(unique_minutes[train_minutes:train_minutes+test_minutes])]

        fX_train = fX[fX['time_interval'].isin(unique_minutes[:train_minutes])]
        fy_train = fy[fX['time_interval'].isin(unique_minutes[:train_minutes])]

        fX_test = fX[fX['time_interval'].isin(unique_minutes[train_minutes:train_minutes+test_minutes])]
        fy_test = fy[fX['time_interval'].isin(unique_minutes[train_minutes:train_minutes+test_minutes])]

        # Drop the 'time_interval' column
        for df in [X_train, X_test, fX_train, fX_test]:
            df.drop(columns=['time_interval'], inplace=True)

        # Print dataset sizes
        print("Size of the training set: ", X_train.shape)
        print("Size of the slow testing set: ", X_test.shape)
        print("Size of the fast training set: ", fX_train.shape)
        print("Size of the fast testing set: ", fX_test.shape)

        return X, y, fX, fy, X_train, y_train, X_test, y_test, fX_train, fy_train, fX_test, fy_test
    

    def evaluate_authenticators(self, users, X_train, y_train, X_test, y_test):
        """
        Evaluate the classifiers using various evaluation metrics.

        Args:
            users (integer): Number of users
            X_train (array-like): Training data features.
            y_train (array-like): Training data labels.
            X_test (array-like): Test data features.
            y_test (array-like): Test data labels.
            fast_X (array-like): Additional data features for fast evaluation.
            fast_y (array-like): Additional data labels for fast evaluation.

        Returns:
            Accuracy_set (DataFrame): DataFrame containing evaluation metrics for each classifier.
            models (list): List of tuples containing the name, accuracy, and model instance.
            best_model_slow (object): Best performing model on the test set.
            best_model_fast (object): Best performing model on the fast set.
            cm_slow (array-like): Confusion matrix of the best performing model on the test set.
            cm_fast (array-like): Confusion matrix of the best performing model on the fast set.
        """
        models = []
        best_model_fast_name = None
        best_model_slow_name = None
        best_model_slow = None
        best_model_fast = None
        cm_slow = None
        cm_fast = None
        Accuracy_set = pd.DataFrame(index=None, columns=[
            'Model', 'Accuracy(Train)', 'Accuracy(Test)', 'Precision(Train)', 'Precision(Test)',
            'Precisions(Train)', 'Precisions(Test)','Accuracies(Train)', 'Accuracies(Test)',
            'Recall(Train)', 'Recall(Test)', 'Recalls(Train)', 'Recalls (Test)'
        ])
        oversampler = SMOTE()
        # Evaluate each individual classifier
        for i in tqdm(range(len(self.classifiers))):
            name = self.classifiers[i][0]
            train_accuracies = []
            test_accuracies = []
            train_precs = []
            test_precs = []
            train_recalls = []
            test_recalls = []
            models.append([])
            for u in range(users):
                model = clone(self.classifiers[i][1])
                y_train_u = np.asarray(y_train == u, dtype='int')
                y_test_u = np.asarray(y_test == u, dtype='int')

                # SMOTE oversampling
                X_smote, y_smote = oversampler.fit_resample(X_train, y_train_u)

                model.fit(X_smote, y_smote)

                y_train_predicted = model.predict(X_smote)
                y_test_predicted = model.predict(X_test)

                accuracy_train = accuracy_score(y_smote, y_train_predicted)
                accuracy_test = accuracy_score(y_test_u, y_test_predicted)

                train_accuracies.append(accuracy_train)
                test_accuracies.append(accuracy_test)

                precision_score_train = precision_score(y_smote, y_train_predicted, average='macro')
                precision_score_test = precision_score(y_test_u, y_test_predicted, average='macro')

                train_precs.append(precision_score_train)
                test_precs.append(precision_score_test)

                recall_score_train = recall_score(y_smote, y_train_predicted, average='macro')
                recall_score_test = recall_score(y_test_u, y_test_predicted, average='macro')

                train_recalls.append(recall_score_train)
                test_recalls.append(recall_score_test)

                # store the models
                models[-1].append((name, accuracy_test, model))

            accuracy_train = np.mean(train_accuracies)
            accuracy_test = np.mean(test_accuracies)
            precision_train = np.mean(train_precs)
            precision_test = np.mean(test_precs)
            recall_train = np.mean(train_recalls)
            recall_test = np.mean(test_recalls)

            Accuracy_set = pd.concat([Accuracy_set, pd.DataFrame({
                'Model': [name],
                'Accuracy(Train)': [accuracy_train],
                'Accuracy(Test)': [accuracy_test],
                'Accuracies(Train)': [train_accuracies],
                'Accuracies(Test)': [test_accuracies],
                'Precision(Train)': [precision_train],
                'Precision(Test)': [precision_test],
                'Precisions(Train)': [train_precs],
                'Precisions(Test)': [test_precs],
                'Recall(Train)': [recall_train],
                'Recall(Test)': [recall_test],
                'Recalls(Train)': [train_recalls],
                'Recalls(Test)': [test_recalls],

            })], ignore_index=True)

        return Accuracy_set, models



    def plot_accuracy_by_vote(self, models, X_test, y_test, increment=1):
        sample_counts = []

        unique_ids = y_test.unique()
        max_samples_per_id = X_test.groupby(y_test).size().min()

        accuracies = np.zeros((max_samples_per_id))
        tp = np.zeros((max_samples_per_id))
        fp = np.zeros((max_samples_per_id))
        tn = np.zeros((max_samples_per_id))
        fn = np.zeros((max_samples_per_id))

        for num_samples in range(1, max_samples_per_id + 1, increment):
            avg_samples = int(np.floor(max_samples_per_id / num_samples))
            for sample in range(avg_samples):
                for real_user in range(len(unique_ids)):
                    real_id = unique_ids[real_user]
                    X_id = X_test[y_test == real_id].iloc[num_samples * sample:num_samples * (sample + 1)]
                    for auth_user in range(len(unique_ids)):
                        auth_id = unique_ids[auth_user]
                        model = models[1][auth_user][-1]
                        predictions = model.predict(X_id)
                        votes = 0
                        for pred_id in predictions:
                            if (pred_id):
                                votes += 1 / len(predictions)
                        final_prediction = int(votes > 0.5)

                        if (auth_user == real_user):
                            if (final_prediction):
                                tp[num_samples - 1] += 1
                            else:
                                fn[num_samples - 1] += 1
                        else:
                            if (final_prediction):
                                fp[num_samples - 1] += 1
                            else:
                                tn[num_samples - 1] += 1
                accuracies[num_samples - 1] += (tp[num_samples - 1] + tn[num_samples - 1]) / (tp[num_samples - 1] + tn[num_samples - 1] + fp[num_samples - 1] + fn[num_samples - 1]) / avg_samples
            sample_counts.append(num_samples * 10)
            sample_counts.append(num_samples * 10)

        sample_counts = sample_counts[::2]
        accuracies = accuracies[::2]
        tp = tp[::2]
        fp = fp[::2]
        tn = tn[::2]
        fn = fn[::2]

        print(accuracies, tp, tn, fp, fn)
        # Plotting
        plt.figure(figsize=(25, 10))
        plt.plot(sample_counts, accuracies, marker='o', linestyle='-', label="Voting Accuracy")
        plt.title('Accuracy vs. Number of Voting Samples')
        plt.xlabel('Number of Seconds in Voting Set')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.figure(figsize=(25, 10))
        plt.plot(sample_counts, fn / (fn + tn), marker='o', linestyle='-', label="Voting Accuracy")
        plt.title('False Negative Rate vs. Number of Voting Samples')
        plt.xlabel('Number of Seconds in Voting Set')
        plt.ylabel('False Negative Rate')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Create and display table
        table_data = {'Seconds in Voting Set': sample_counts, 'Voting Accuracy': accuracies}
        df = pd.DataFrame(table_data)
        print("\nVote Over Time Data Table:")
        display(df)
