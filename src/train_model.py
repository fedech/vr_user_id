from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
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
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.cluster import Birch
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, log_loss, confusion_matrix
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import VotingClassifier

class ClassifierEvaluator:
    """
    A class to evaluate multiple classifiers using various evaluation metrics.

    Attributes:
        classifiers (list): A list containing the names and instances of classifiers to be evaluated.
        SEED (int): Random seed for reproducibility.
    """

    def __init__(self, SEED=42):
        self.classifiers = [
            ['SVC', SVC(kernel="rbf", C=0.025, probability=True, random_state=SEED)],
            ['ExtraTreesClassifier', ExtraTreesClassifier(bootstrap=False, max_depth=None,
                                                          min_samples_split=2, n_estimators=800,
                                                          random_state=SEED)],
            ["LinearDiscriminantAnalysis", LinearDiscriminantAnalysis()],
            ['DecisionTreeClassifier', DecisionTreeClassifier(random_state=SEED)],
            ['KNeighborsClassifier', KNeighborsClassifier()],
            ['RandomForestClassifier', RandomForestClassifier(random_state=SEED)],
            ['MLPClassifier', MLPClassifier(random_state=SEED)],
            ['AdaBoostClassifier', AdaBoostClassifier(random_state=SEED)],
            ['GaussianNB', GaussianNB()],
            ['QuadraticDiscriminantAnalysis', QuadraticDiscriminantAnalysis(store_covariance=True)],
            ['LogisticRegression', LogisticRegression(random_state=SEED)],
            ['BernoulliNB', BernoulliNB()],
            ['BaggingClassifier', BaggingClassifier(random_state=SEED)],
            ['LGBMClassifier', LGBMClassifier(random_state=SEED, verbosity=-1)]
        ]
        
        # Add the VotingClassifier
        voting_classifiers = [
            ('ET', self.classifiers[1][1]),  # ExtraTreesClassifier
            ('LGBM', self.classifiers[-1][1]),  # LGBMClassifier
            ('LR', self.classifiers[10][1]),  # LogisticRegression
            ('QDA', self.classifiers[9][1]),  # QuadraticDiscriminantAnalysis
            ('RF', self.classifiers[5][1])  # RandomForestClassifier
        ]
        
        voting_classifier = VotingClassifier(
            estimators=voting_classifiers,
            voting='soft',
            n_jobs=-1
        )
        
        self.classifiers.append(['VotingClassifier', voting_classifier])
        self.clustering_algorithms = [
                            ['KMeans', KMeans(n_clusters=60, random_state=SEED)],
                            ['AgglomerativeClustering', AgglomerativeClustering(n_clusters=60)],
                            ['SpectralClustering', SpectralClustering(n_clusters=60)],
                            ['Birch', Birch(n_clusters=60)]]
        self.SEED = SEED

    def evaluate_clustering(self, X_train):
        """
        Evaluate the clustering algorithms using various evaluation metrics.

        Args:
            X_train (array-like): Training data features.
            y_train (array-like): Training data labels (for semi-supervised learning).
            X_test (array-like): Test data features.
            y_test (array-like): Test data labels (for semi-supervised learning).
            fast_X (array-like): Additional data features for fast evaluation.
            fast_y (array-like): Additional data labels for fast evaluation.
            n_clusters (int): Number of clusters to be formed.

        Returns:
            Evaluation_set (DataFrame): DataFrame containing evaluation metrics for each clustering algorithm.
            models (list): List of tuples containing the name and the model instance.
            best_model (object): Best performing model based on classification metrics.
            cm (array-like): Confusion matrix of the best performing model.
        """
        models = []
        best_score = 0
        best_model = None
        cm = None
        Evaluation_set = pd.DataFrame(index=None, columns=['Model', 'Silhouette', 
                                                        'Calinski_Harabasz',])

        for i in tqdm(range(len(self.clustering_algorithms))):
            name = self.clustering_algorithms[i][0]
            model = self.clustering_algorithms[i][1]

            model.fit(X_train)

            y_train_predicted = model.fit_predict(X_train)

            silhouette = silhouette_score(X_train, y_train_predicted)

            calinski_harabasz = calinski_harabasz_score(X_train, y_train_predicted)

            # Store the models
            models.append((name, model))

            if calinski_harabasz > best_score:
                best_model = model
                best_score = calinski_harabasz

            Evaluation_set = pd.concat([Evaluation_set, pd.DataFrame({'Model': [name], 'Silhouette': [silhouette],
                                                          'Calinski_Harabasz': [calinski_harabasz]})], 
                           ignore_index=True)

        return Evaluation_set, models, best_model, cm
    
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
    

    def evaluate_classifiers(self, X_train, y_train, X_test, y_test, fast_X, fast_y):
        """
        Evaluate the classifiers using various evaluation metrics.

        Args:
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
        best_score_slow = 0
        best_score_fast = 0
        best_model_fast_name = None
        best_model_slow_name = None
        best_model_slow = None
        best_model_fast = None
        cm_slow = None
        cm_fast = None
        Accuracy_set = pd.DataFrame(index=None, columns=[
            'Model', 'Accuracy(Train)', 'Accuracy(Slow)', 'Accuracy(Fast)', 
            'F1(Train)', 'F1(Slow)', 'F1(Fast)',
            'Precision(Train)', 'Precision(Slow)', 'Precision(Fast)',
            'Recall(Train)', 'Recall(Slow)', 'Recall(Fast)',
            'Log_loss(Train)', 'Log_loss(Slow)', 'Log_loss(Fast)'
        ])

        # Evaluate each individual classifier
        for i in tqdm(range(len(self.classifiers))):
            name = self.classifiers[i][0]
            model = self.classifiers[i][1]

            model.fit(X_train, y_train)

            y_train_predicted = model.predict(X_train)
            y_test_predicted = model.predict(X_test)
            y_fast_predicted = model.predict(fast_X)

            accuracy_train = accuracy_score(y_train, y_train_predicted)
            accuracy_test = accuracy_score(y_test, y_test_predicted)
            accuracy_fast = accuracy_score(fast_y, y_fast_predicted)

            f1_score_train = f1_score(y_train, y_train_predicted, average='macro')
            f1_score_test = f1_score(y_test, y_test_predicted, average='macro')
            f1_score_fast = f1_score(fast_y, y_fast_predicted, average='macro')

            precision_score_train = precision_score(y_train, y_train_predicted, average='macro')
            precision_score_test = precision_score(y_test, y_test_predicted, average='macro')
            precision_score_fast = precision_score(fast_y, y_fast_predicted, average='macro')

            recall_score_train = recall_score(y_train, y_train_predicted, average='macro')
            recall_score_test = recall_score(y_test, y_test_predicted, average='macro')
            recall_score_fast = recall_score(fast_y, y_fast_predicted, average='macro')

            log_loss_train = log_loss(y_train, model.predict_proba(X_train))
            log_loss_test = log_loss(y_test, model.predict_proba(X_test))
            log_loss_fast = log_loss(fast_y, model.predict_proba(fast_X))

            # store the models
            models.append((name, accuracy_test, model))

            if accuracy_test > best_score_slow and name not in ['VotingClassifier', "LinearDiscriminantAnalysis", "QuadraticDiscriminantAnalysis"]:
                best_model_slow = model
                best_score_slow = accuracy_test
                best_model_slow_name = i
                cm_slow = confusion_matrix(y_test, y_test_predicted)

            if accuracy_fast > best_score_fast and name not in ['VotingClassifier', "LinearDiscriminantAnalysis", "QuadraticDiscriminantAnalysis"]:
                best_model_fast = model
                best_score_fast = accuracy_fast
                best_model_fast_name = i
                cm_fast = confusion_matrix(fast_y, y_fast_predicted)

            Accuracy_set = pd.concat([Accuracy_set, pd.DataFrame({
                'Model': [name],
                'Accuracy(Train)': [accuracy_train],
                'Accuracy(Slow)': [accuracy_test],
                'Accuracy(Fast)': [accuracy_fast],
                'F1(Train)': [f1_score_train],
                'F1(Slow)': [f1_score_test],
                'F1(Fast)': [f1_score_fast],
                'Precision(Train)': [precision_score_train],
                'Precision(Slow)': [precision_score_test],
                'Precision(Fast)': [precision_score_fast],
                'Recall(Train)': [recall_score_train],
                'Recall(Slow)': [recall_score_test],
                'Recall(Fast)': [recall_score_fast],
                'Log_loss(Train)': [log_loss_train],
                'Log_loss(Slow)': [log_loss_test],
                'Log_loss(Fast)': [log_loss_fast]
            })], ignore_index=True)

        return Accuracy_set, models, best_model_slow, best_model_fast, cm_slow, cm_fast, best_model_slow_name, best_model_fast_name



    def plot_accuracy_by_vote(self, model, X_test, y_test, increment=1):
        accuracies = []
        sample_counts = []

        unique_ids = y_test.unique()
        max_samples_per_id = X_test.groupby(y_test).size().min()

        for num_samples in range(1, max_samples_per_id + 1, increment):
            votes = {id: {} for id in unique_ids}

            for id in unique_ids:
                X_id = X_test[y_test == id].iloc[:num_samples]
                predictions = model.predict(X_id)
                
                for pred_id in predictions:
                    votes[id][pred_id] = votes[id].get(pred_id, 0) + 1

            final_predictions = {id: max(id_votes, key=id_votes.get) for id, id_votes in votes.items()}

            correct = sum(final_predictions[id] == id for id in unique_ids)
            accuracy = correct / len(unique_ids)

            accuracies.append(accuracy)
            sample_counts.append(num_samples * 10)

        # Plotting
        plt.figure(figsize=(25, 10))
        plt.plot(sample_counts, accuracies, marker='o', linestyle='-', label="Voting Accuracy")
        plt.title('Accuracy vs. Number of Voting Samples')
        plt.xlabel('Number of Seconds in Voting Set')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Create and display table
        table_data = {'Seconds in Voting Set': sample_counts, 'Voting Accuracy': accuracies}
        df = pd.DataFrame(table_data)
        print("\nVote Over Time Data Table:")
        display(df)

    def accuracy_over_time(self, X, y, best_model, unique_minutes, X_test, y_test, fast_X, fast_y):
        accuracies_slow = []
        accuracies_fast = []

        for max_minute in unique_minutes:
            selected_minutes = unique_minutes[:max_minute]
            X_train = X[X['time_interval'].isin(selected_minutes)]
            y_train = y[X['time_interval'].isin(selected_minutes)]
            
            X_train_filtered = X_train.drop(columns=['time_interval'])
            
            model = self.classifiers[best_model][1]
            model.fit(X_train_filtered, y_train)
            
            y_pred = model.predict(X_test)
            y_pred_fast = model.predict(fast_X)
            accuracies_slow.append(accuracy_score(y_test, y_pred))
            accuracies_fast.append(accuracy_score(fast_y, y_pred_fast))
        
        # Plotting
        plt.figure(figsize=(25, 10))
        plt.plot(np.array(unique_minutes)*60, accuracies_slow, marker='o', linestyle='-', label="Slow Game")
        plt.plot(np.array(unique_minutes)*60, accuracies_fast, marker='o', linestyle='-', label="Fast Game")
        plt.title('Testing Set Accuracy vs. Number of Training Seconds')
        plt.xlabel('Number of Seconds in Training Set')
        plt.ylabel('Accuracy on Testing Set')
        plt.legend()
        plt.show()

        # Create and display table
        table_data = {
            'Seconds in Training Set': np.array(unique_minutes)*60,
            'Slow Game Accuracy': accuracies_slow,
            'Fast Game Accuracy': accuracies_fast
        }
        df = pd.DataFrame(table_data)
        print("\nAccuracy Over Time Data Table:")
        display(df)

    def accuracy_over_ids(self, X, y, best_model, X_test, y_test, fast_X, fast_y):
        unique_ids = np.sort(y.unique())
        accuracies_slow = []
        accuracies_fast = []
        id_ranges = range(5, len(unique_ids)+1, 5)

        for num_ids in id_ranges:
            selected_ids = unique_ids[:num_ids]
            model = self.classifiers[best_model][1]
            model.fit(X[y.isin(selected_ids)], y[y.isin(selected_ids)])
            
            y_pred = model.predict(X_test[y_test.isin(selected_ids)])
            accuracy_slow = accuracy_score(y_test[y_test.isin(selected_ids)], y_pred)
            accuracies_slow.append(accuracy_slow)

            y_pred_fast = model.predict(fast_X[fast_y.isin(selected_ids)])
            accuracy_fast = accuracy_score(fast_y[fast_y.isin(selected_ids)], y_pred_fast)
            accuracies_fast.append(accuracy_fast)
        
        # Plotting
        plt.figure(figsize=(25, 10))
        plt.plot(id_ranges, accuracies_slow, marker='o', linestyle='-', label="Slow Game")
        plt.plot(id_ranges, accuracies_fast, marker='o', linestyle='-', label="Fast Game")
        plt.title('Testing Set Accuracy vs. Number of IDs in Training Set')
        plt.xlabel('Number of IDs in Training Set')
        plt.ylabel('Accuracy on Testing Set')
        plt.legend()
        plt.show()

        # Create and display table
        table_data = {
            'Number of IDs in Training Set': list(id_ranges),
            'Slow Game Accuracy': accuracies_slow,
            'Fast Game Accuracy': accuracies_fast
        }
        df = pd.DataFrame(table_data)
        print("\nID Over Time Data Table:")
        display(df)