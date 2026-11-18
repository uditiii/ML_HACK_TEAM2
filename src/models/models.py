from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier

def get_models():
    """(name, family, model, config, max_train_rows)"""
    return [
        ("hist_gb", "Boosting", HistGradientBoostingClassifier(class_weight="balanced", random_state=42),
         {"class_weight": "balanced"}, None),
        ("decision_tree", "Tree", DecisionTreeClassifier(max_depth=10, class_weight="balanced", random_state=42),
         {"max_depth": 10, "class_weight": "balanced"}, None),
        ("knn", "Instance-based", Pipeline([("scaler", StandardScaler()),
         ("knn", KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=-1))]),
         {"n_neighbors": 15, "weights": "distance", "train_subsample": 100000}, 100000),
        ("logreg", "Linear", Pipeline([("scaler", StandardScaler()),
         ("lr", LogisticRegression(max_iter=1000, class_weight="balanced"))]),
         {"C": 1.0, "class_weight": "balanced"}, None),
        ("gaussian_nb", "Probabilistic", GaussianNB(), {}, None),
        ("random_forest", "Ensemble", RandomForestClassifier(n_estimators=200, max_depth=12,
         class_weight="balanced", n_jobs=-1, random_state=42), {"n_estimators": 200, "max_depth": 12}, None),
        ("extra_trees", "Ensemble", ExtraTreesClassifier(n_estimators=200, max_depth=12,
         class_weight="balanced", n_jobs=-1, random_state=42), {"n_estimators": 200, "max_depth": 12}, None),
    ]