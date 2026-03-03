import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from sklearn.model_selection import train_test_split

df = pd.read_csv("dataset.csv")

y_class = (df["Rings"] > 9).astype(int)

df = pd.get_dummies(df, columns=["Sex"], drop_first=True)

X = df.drop(columns=["Rings"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y_class, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=50,
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)

explainer = ClassifierExplainer(
    model,
    X_test,
    y_test,
    labels=["Young", "Old"]
)

db = ExplainerDashboard(explainer)
db.to_yaml(
    "dashboard.yaml",
    explainerfile="explainer.joblib",
    dump_explainer=True
)