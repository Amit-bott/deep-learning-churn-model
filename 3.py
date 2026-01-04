import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
 from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

df = pd.read_csv("Churn_Modelling.csv")

df.drop(["RowNumber", "CustomerId", "Surname"], axis=1, inplace=True)

df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
df = pd.get_dummies(df, columns=["Geography"], drop_first=True)

X = df.drop("Exited", axis=1)
y = df["Exited"]

# -------------------------
# SPLIT DATA
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

joblib.dump(scaler, "models/scaler.pkl")

# -------------------------
# ANN MODEL
# # -------------------------
# model = Sequential([
#     Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
#     Dropout(0.3),
#     Dense(32, activation="relu"),
#     Dropout(0.2),
#     Dense(1, activation="sigmoid")
# ])

# # model.compile(
#     optimizer="adam",
#     loss="binary_crossentropy",
#     metrics=["accuracy"]
# )

# early_stop = EarlyStopping(
    # monitor="val_loss",
    # patience=5,
    # restore_best_weights=True
# )

# model.fit(
#     X_train,
#     y_train,
#     validation_split=0.1,
#     epochs=50,
#     batch_size=32,
#     callbacks=[early_stop],
#     verbose=1
# )

# -------------------------
# SAVE MODEL
# -------------------------
# model.save("models/churn_model.h5")

# -------------------------
# EVALUATION
# -------------------------
# y_pred = (model.predict(X_test) > 0.5).astype(int)
# acc = accuracy_score(y_test, y_pred)

print("✅ Model trained successfully")
# print(f"🎯 Accuracy: {acc:.2%}")
