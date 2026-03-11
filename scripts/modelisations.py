from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB
import joblib


from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import os

class Modelisations :

    # Contient les retours des fonctions d'évaluation des modèles 
    def __init__(self, lr: list = [],rs : int = 42) :
        self.list_returns = lr,
        self.random_state = rs

        
    def train_evaluate_logistic_regression(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1
    ) -> dict:

        param_grid_lr = {
            'C': [0.1, 1, 10],
            'penalty': ['l2'],
            'solver': ['lbfgs', 'liblinear'],
            'max_iter': [1000]
        }

        lr = LogisticRegression(random_state=self.random_state)

        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)

        grid_lr = GridSearchCV(
            estimator=lr,
            param_grid=param_grid_lr,
            cv=cv_strategy,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose
        )

        grid_lr.fit(X_train, y_train)

        print(f"\nMeilleurs paramètres : {grid_lr.best_params_}")
        print(f"Meilleur score CV : {grid_lr.best_score_:.4f}")

        y_pred = grid_lr.predict(X_test)
        y_proba = grid_lr.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        result =  {
            "grid" : grid_lr,
            "grid_name" : "logistic_regression",
            "y_pred": y_pred,
            "y_proba": y_proba,
            "accuracy": acc,
            "f1_score": f1,
            "roc_auc": auc
        }

        self.list_returns.append(result)
        



    def train_evaluate_svm(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        sample_size=10000,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1
    ) -> dict:
        """
        Entraîne un SVM avec GridSearchCV sur un sous-échantillon
        puis évalue le meilleur modèle sur le jeu de test.
        """

        param_grid_svm = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }

        svm = SVC(random_state=self.random_state, probability=True)

        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)

        grid_svm = GridSearchCV(
            estimator=svm,
            param_grid=param_grid_svm,
            cv=cv_strategy,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose
        )

        grid_svm.fit(X_train, y_train)

        print(f"\nMeilleurs paramètres : {grid_svm.best_params_}")
        print(f"Meilleur score CV : {grid_svm.best_score_:.4f}")

        # Prédictions
        y_pred = grid_svm.predict(X_test)
        y_proba = grid_svm.predict_proba(X_test)[:, 1]

        # Métriques
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)


        result = {
            "grid" : grid_svm,
            "grid_name" : "svm",
            "y_pred": y_pred,
            "y_proba": y_proba,
            "accuracy": acc,
            "f1_score": f1,
            "roc_auc": auc
        }

        self.list_returns.append(result)


    def train_evaluate_random_forest(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1
    ) -> dict:
        """
        Entraîne un Random Forest avec GridSearchCV
        puis évalue le meilleur modèle sur le jeu de test.
        """

        param_grid_rf = {
            'n_estimators': [100, 200],
            'max_depth': [20, 30, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }

        rf = RandomForestClassifier(
            random_state=self.random_state,
            n_jobs=n_jobs
        )

        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)

        grid_rf = GridSearchCV(
            estimator=rf,
            param_grid=param_grid_rf,
            cv=cv_strategy,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose
        )

        grid_rf.fit(X_train, y_train)

        print(f"\nMeilleurs paramètres : {grid_rf.best_params_}")
        print(f"Meilleur score CV : {grid_rf.best_score_:.4f}")

        # Prédictions
        y_pred = grid_rf.predict(X_test)
        y_proba = grid_rf.predict_proba(X_test)[:, 1]

        # Métriques
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)



        result =  {
            "grid" : grid_rf,
            "grid_name" : "random_forest",
            "y_pred": y_pred,
            "y_proba": y_proba,
            "accuracy": acc,
            "f1_score": f1,
            "roc_auc": auc
        }

        self.list_returns.append(result)

    def train_evaluate_voting_classifier(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        voting="soft",
        n_jobs=-1
    ) -> dict:
        """
        Crée un Voting Classifier à partir des meilleurs modèles
        Logistic Regression et Random Forest + Naive Bayes.
        Puis évalue le modèle sur le jeu de test.
        """
        list_estimators = []
        for best in self.list_returns :
            list_estimators.append((best["grid_name"],best["grid"]))

        # Naive Bayes (très efficace avec TF-IDF)
        nb = MultinomialNB()
        nb.fit(X_train, y_train)

        list_estimators.append(('nb', nb))
        
        # Voting Classifier
        voting_clf = VotingClassifier(
            estimators=list_estimators,
            voting=voting,
            n_jobs=n_jobs
        )

        voting_clf.fit(X_train, y_train)

        # Prédictions
        y_pred = voting_clf.predict(X_test)
        y_proba = voting_clf.predict_proba(X_test)[:, 1]

        # Métriques
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        
        return {
            "model": voting_clf,
            "accuracy": acc,
            "f1_score": f1,
            "roc_auc": auc,
            "y_pred": y_pred,
            "y_proba": y_proba
        }
    

    def list_results_models(self) :
        return self.list_returns