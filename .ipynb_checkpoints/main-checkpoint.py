# GUI
from tkinter import messagebox, Text, END, Label, Scrollbar
from tkinter import filedialog
import tkinter as tk

import pandas as pd
# Classification models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# Metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Core libraries
import os
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing & splitting
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from sklearn.ensemble import StackingRegressor
import lightgbm as lgb

import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk

from tkinter import simpledialog, messagebox
import os
import joblib
import tkinter as tk
from tkinter import filedialog, Text, ttk
from PIL import Image, ImageTk

warnings.filterwarnings('ignore')

global MODEL_DIR, filename, X, Y, model, categories


global MODEL_DIR
global filename
global X, Y
global model
global categories


def upload_dataset():
    global df
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    df = pd.read_csv(file_path)

    text.delete('1.0', END)
    text.insert(END, "First 5 rows of the dataset:\n\n")
    text.insert(END, df.head())
    text.insert(END, "\n\n")
    text.insert(END, f"Shape: Rows = {df.shape[0]}, Columns = {df.shape[1]}\n")


    
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


# ---------------- Preprocessing (Training / Prediction) ----------------
def preprocess_data_tkinter():
    global X, Y, df
    text.delete('1.0', END)  # Clear previous text

    # Drop ID column if exists
    if 'ID' in df.columns:
        df.drop(columns=['ID'], inplace=True)
        text.insert(END, "Dropped ID column.\n")

    # Encode categorical columns
    categorical_cols = [
        'Soil_Type', 'Crop_Type', 'Crop_Growth_Stage', 'Season',
        'Irrigation_Type', 'Previous_Crop', 'Region', 'Fertilizer_Used_Last_Season'
    ]
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        joblib.dump(le, os.path.join(MODEL_DIR, f"{col}_encoder.pkl"))
        text.insert(END, f"Encoded {col} and saved encoder.\n")

    # Feature matrix
    X = df.drop(columns=['Recommended_Fertilizer'])

    # Numeric columns scaling
    numeric_cols = [
        'Soil_pH', 'Soil_Moisture', 'Organic_Carbon', 'Electrical_Conductivity',
        'Nitrogen_Level', 'Phosphorus_Level', 'Potassium_Level',
        'Temperature', 'Humidity', 'Rainfall', 'Yield_Last_Season'
    ]
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'standard_scaler.pkl'))
    text.insert(END, "Scaled numeric columns and saved scaler.\n")

    # Target column
    Y = df['Recommended_Fertilizer']

    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_res, Y_res = smote.fit_resample(X, Y)
    X, Y = X_res, Y_res
    text.insert(END, f"Applied SMOTE. Original samples: {len(df)}, New samples: {len(Y_res)}\n")
    text.insert(END, "Preprocessing complete.\n")

def perform_eda_tkinter():
    global df
    text.delete('1.0', END)

    plt.figure(figsize=(18, 12))

    # Plot 1: Target Distribution
    plt.subplot(2, 3, 1)
    sns.countplot(x='Recommended_Fertilizer', data=df)
    plt.title('Distribution of Recommended Fertilizer')

    # Plot 2: Soil Type vs Recommended Fertilizer
    plt.subplot(2, 3, 2)
    sns.countplot(x='Soil_Type', hue='Recommended_Fertilizer', data=df)
    plt.title('Soil Type vs Recommended Fertilizer')

    # Plot 3: Crop Type vs Recommended Fertilizer
    plt.subplot(2, 3, 3)
    sns.countplot(x='Crop_Type', hue='Recommended_Fertilizer', data=df)
    plt.title('Crop Type vs Recommended Fertilizer')

    # Plot 4: Soil pH vs Yield_Last_Season
    plt.subplot(2, 3, 4)
    sns.scatterplot(x='Soil_pH', y='Yield_Last_Season', hue='Recommended_Fertilizer', data=df)
    plt.title('Soil pH vs Last Season Yield')

    # Plot 5: Nitrogen Level vs Yield_Last_Season
    plt.subplot(2, 3, 5)
    sns.scatterplot(x='Nitrogen_Level', y='Yield_Last_Season', hue='Recommended_Fertilizer', data=df)
    plt.title('Nitrogen Level vs Last Season Yield')

    # Plot 6: Season vs Recommended Fertilizer
    plt.subplot(2, 3, 6)
    sns.countplot(x='Season', hue='Recommended_Fertilizer', data=df)
    plt.title('Season vs Recommended Fertilizer')

    plt.tight_layout()
    plt.savefig('fertilizer_eda_plots.png')
    plt.show()

    text.insert(END, "EDA plots saved as 'fertilizer_eda_plots.png'\n")

def split_train_test_tkinter():
    global X_train, X_test, y_train, y_test, X, Y
    text.delete('1.0', END)

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    text.insert(END, f"Train/Test Split Completed:\n")
    text.insert(END, f"X_train: {X_train.shape}, X_test: {X_test.shape}\n")
    text.insert(END, f"y_train: {y_train.shape}, y_test: {y_test.shape}\n")


# Global DataFrame for metrics
classification_metrics_df = pd.DataFrame(
    columns=['Algorithm', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
)

# Tkinter Text widget
text = None  # Will be initialized in GUI

# -----------------------------
# Metrics function
# -----------------------------
def calculate_metrics_tkinter(task_type, algorithm, y_pred, y_test, y_score=None):
    global classification_metrics_df, text

    # Load target label encoder if exists
    encoder_path = os.path.join("models", "label_encoder.pkl")
    if os.path.exists(encoder_path):
        le_target = joblib.load(encoder_path)
        categories = le_target.classes_
    else:
        categories = np.unique(y_test)

    n_classes = len(categories)

    # Classification metrics
    acc = accuracy_score(y_test, y_pred) * 100
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0) * 100
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0) * 100
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0) * 100

    classification_metrics_df.loc[len(classification_metrics_df)] = [
        algorithm, acc, prec, rec, f1
    ]

    text.delete('1.0', END)
    text.insert(END, f"{algorithm} Metrics\n")
    text.insert(END, "-"*40 + "\n")
    text.insert(END, f"Accuracy : {acc:.2f}%\n")
    text.insert(END, f"Precision: {prec:.2f}%\n")
    text.insert(END, f"Recall   : {rec:.2f}%\n")
    text.insert(END, f"F1-Score : {f1:.2f}%\n\n")

    report = classification_report(
        y_test,
        y_pred,
        labels=range(n_classes),
        target_names=categories,
        zero_division=0
    )
    text.insert(END, "Classification Report:\n")
    text.insert(END, report + "\n")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=categories,
        yticklabels=categories
    )
    plt.title(f"{algorithm} - Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(f"results/{algorithm.replace(' ', '_')}_confusion_matrix.png")
    plt.show()

    # ROC curve
    if y_score is not None:
        plt.figure(figsize=(10, 8))

        if n_classes == 2:
            fpr, tpr, _ = roc_curve(y_test, y_score[:, 1])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.2f}")
        else:
            y_test_bin = label_binarize(y_test, classes=range(n_classes))
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, label=f"{categories[i]} (AUC = {roc_auc:.2f})")

        plt.plot([0, 1], [0, 1], "k--", lw=1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{algorithm} - ROC Curve")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"results/{algorithm.replace(' ', '_')}_roc_curve.png")
        plt.show()

    text.insert(END, f"{algorithm} metrics and plots saved.\n")

      
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import os
import joblib
from tkinter import END

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def train_lda_classifier_tkinter(X_train, y_train, X_test, y_test, solver='svd', shrinkage=None):
    global text

    model_path = os.path.join(MODEL_DIR, 'lda.pkl')
    text.delete('1.0', END)

    if os.path.exists(model_path):
        text.insert(END, "Loading LDA Classifier...\n")
        model = joblib.load(model_path)
    else:
        text.insert(END, "Training LDA Classifier...\n")
        model = LinearDiscriminantAnalysis(
            solver=solver,
            shrinkage=shrinkage
        )
        model.fit(X_train, y_train)
        joblib.dump(model, model_path)
        text.insert(END, f"Model saved to {model_path}\n")

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)
    else:
        y_score = None

    calculate_metrics_tkinter(
        task_type='classification',
        algorithm="Linear Discriminant Analysis",
        y_pred=y_pred,
        y_test=y_test,
        y_score=y_score
    )

    text.insert(END, "LDA training and evaluation completed.\n")
    return model




from sklearn.svm import SVC
import os
import joblib
from tkinter import END

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def train_kernel_svm_classifier_tkinter(
    X_train,
    y_train,
    X_test,
    y_test,
    kernel='rbf',
    C=1.0,
    gamma='scale'
):
    global text

    model_path = os.path.join(MODEL_DIR, 'kernel_svm.pkl')
    text.delete('1.0', END)

    if os.path.exists(model_path):
        text.insert(END, "Loading Kernel SVM Classifier...\n")
        model = joblib.load(model_path)
    else:
        text.insert(END, "Training Kernel SVM Classifier...\n")
        model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            probability=True,
            random_state=42
        )
        model.fit(X_train, y_train)
        joblib.dump(model, model_path)
        text.insert(END, f"Model saved to {model_path}\n")

    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)

    calculate_metrics_tkinter(
        task_type='classification',
        algorithm="Kernel SVM",
        y_pred=y_pred,
        y_test=y_test,
        y_score=y_score
    )

    text.insert(END, "Kernel SVM training and evaluation completed.\n")
    return model


from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import os
import joblib
import numpy as np
from tkinter import END

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def train_qda_classifier_tkinter(
    X_train,
    y_train,
    X_test,
    y_test,
    reg_param=0.0
):
    global text

    model_path = os.path.join(MODEL_DIR, 'qda.pkl')
    text.delete('1.0', END)

    # Convert pandas → numpy (safe for QDA)
    X_train = np.asarray(X_train, dtype=np.float64)
    X_test = np.asarray(X_test, dtype=np.float64)

    if os.path.exists(model_path):
        text.insert(END, "Loading QDA Classifier...\n")
        model = joblib.load(model_path)
    else:
        text.insert(END, "Training QDA Classifier...\n")
        model = QuadraticDiscriminantAnalysis(reg_param=reg_param)
        model.fit(X_train, y_train)
        joblib.dump(model, model_path)
        text.insert(END, f"Model saved to {model_path}\n")

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)
    else:
        y_score = None

    calculate_metrics_tkinter(
        task_type='classification',
        algorithm="Quadratic Discriminant Analysis",
        y_pred=y_pred,
        y_test=y_test,
        y_score=y_score
    )

    text.insert(END, "QDA training and evaluation completed.\n")
    return model


import os
import joblib
from sklearn.ensemble import BaggingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

# Global MODEL_DIR
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Global Text widget
text = None  # Initialized in GUI

def train_bagging_logistic_stacking_tkinter(X_train, y_train, X_test, y_test, n_estimators=100):
    global text

    model_path = os.path.join(MODEL_DIR, 'bagging_logistic_stacking1.pkl')

    text.delete('1.0', END)

    if os.path.exists(model_path):
        text.insert(END, "Loading existing Stacking (Bagging + Logistic) model...\n")
        model = joblib.load(model_path)
    else:
        text.insert(END, "Training Stacking (Bagging + Logistic) model...\n")

        # Base Bagging model
        bagging_model = BaggingClassifier(
            n_estimators=n_estimators,
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        )

        # Stacking with Logistic Regression
        model = StackingClassifier(
            estimators=[('bagging', bagging_model)],
            final_estimator=LogisticRegression(max_iter=1000, random_state=42),
            cv=5,
            passthrough=True,
            n_jobs=-1
        )

        model.fit(X_train, y_train)
        joblib.dump(model, model_path)
        text.insert(END, f"Model trained and saved to {model_path}\n")

    # Predictions
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)

    # Evaluation using Tkinter version of metrics
    calculate_metrics_tkinter(
        task_type='classification',
        algorithm="Stacking",
        y_pred=y_pred,
        y_test=y_test,
        y_score=y_score
    )

    text.insert(END, "Stacking model training and evaluation complete.\n")
    return model


def plot_model_performance_tkinter():
    text.delete('1.0', END)
    text.insert(END, "Plotting Model Performance Comparison...\n")

    sns.set(style="whitegrid")
    plt.figure(figsize=(14, 7))

    df_melt = classification_metrics_df.melt(
        id_vars='Algorithm',
        value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        var_name='Metric',
        value_name='Score'
    )

    palette_colors = {
        'Accuracy': '#1f77b4',  
        'Precision': '#ff7f0e',  
        'Recall': '#2ca02c',     
        'F1-Score': '#d62728'    
    }

    ax = sns.barplot(x='Algorithm', y='Score', hue='Metric', data=df_melt, palette=palette_colors)

    plt.xticks(rotation=45, ha='right')
    plt.title("Model Performance Comparison")
    plt.ylabel("Score (%)")
    plt.ylim(0, 105) 

    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f"{height:.1f}", 
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=10, color='black', xytext=(0, 2),
                    textcoords='offset points')

    plt.legend(title="Metric")
    plt.tight_layout()
    plt.savefig("results/model_performance_comparison.png")
    plt.show()

    text.insert(END, "Plot saved as 'results/model_performance_comparison.png'\n")


    
import os
import joblib
import numpy as np
import pandas as pd
from tkinter import END, filedialog

MODEL_DIR = "models"

# -----------------------------
# Preprocess test data (Tkinter)
# -----------------------------
def preprocess_test_data_tkinter(test_df):
    test_df = test_df.copy()

    if 'ID' in test_df.columns:
        test_df.drop(columns=['ID'], inplace=True)

    categorical_cols = [
        'Soil_Type', 'Crop_Type', 'Crop_Growth_Stage', 'Season',
        'Irrigation_Type', 'Previous_Crop', 'Region',
        'Fertilizer_Used_Last_Season'
    ]

    for col in categorical_cols:
        encoder_path = os.path.join(MODEL_DIR, f"{col}_encoder.pkl")
        le = joblib.load(encoder_path)

        test_df[col] = test_df[col].astype(str)
        unseen = set(test_df[col].unique()) - set(le.classes_)
        if unseen:
            test_df[col] = test_df[col].apply(
                lambda x: x if x in le.classes_ else le.classes_[0]
            )

        test_df[col] = le.transform(test_df[col])

    numeric_cols = [
        'Soil_pH', 'Soil_Moisture', 'Organic_Carbon', 'Electrical_Conductivity',
        'Nitrogen_Level', 'Phosphorus_Level', 'Potassium_Level',
        'Temperature', 'Humidity', 'Rainfall', 'Yield_Last_Season'
    ]

    scaler = joblib.load(os.path.join(MODEL_DIR, 'standard_scaler.pkl'))
    test_df[numeric_cols] = scaler.transform(test_df[numeric_cols])

    return test_df


# -----------------------------
# Predict using Stacking Model
# -----------------------------
def predict_stacking_model_tkinter():
    global test_df

    text.delete('1.0', END)

    model = joblib.load(os.path.join(MODEL_DIR, 'bagging_logistic_stacking1.pkl'))

    X_test = preprocess_test_data_tkinter(test_df)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    confidence = np.max(y_proba, axis=1)

    results = test_df.copy()
    results['Predicted_Fertilizer'] = y_pred
    results['Confidence'] = confidence

    text.insert(END, "Prediction Results (First 10 Rows):\n\n")
    text.insert(END, results.head(10))
    text.insert(END, "\n\nPrediction completed successfully.\n")


# -----------------------------
# Upload Test Data
# -----------------------------
def upload_test_data():
    global test_df
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    test_df = pd.read_csv(file_path)

    text.delete('1.0', END)
    text.insert(END, "Test dataset loaded\n")
    text.insert(END, test_df.head())
    text.insert(END, f"\n\nShape: {test_df.shape}\n")



def close():
    main.destroy()
    

# Predefined credentials
ADMIN_CREDENTIALS = {"username": "admin", "password": "admin"}
USER_CREDENTIALS  = {"username": "user", "password": "user"}

def authenticate(role):
    login_win = tk.Toplevel(main)
    login_win.title(f"{role} Login")
    login_win.geometry("300x200")
    login_win.grab_set()

    tk.Label(login_win, text="Username:").pack(pady=5)
    username_entry = tk.Entry(login_win)
    username_entry.pack(pady=5)

    tk.Label(login_win, text="Password:").pack(pady=5)
    password_entry = tk.Entry(login_win, show="*")
    password_entry.pack(pady=5)

    def check_login():
        username = username_entry.get()
        password = password_entry.get()

        if role == "ADMIN":
            if username == ADMIN_CREDENTIALS["username"] and password == ADMIN_CREDENTIALS["password"]:
                login_win.destroy()
                show_admin_buttons()
            else:
                messagebox.showerror("Error", "Invalid Admin credentials!")
        elif role == "USER":
            if username == USER_CREDENTIALS["username"] and password == USER_CREDENTIALS["password"]:
                login_win.destroy()
                show_user_buttons()
            else:
                messagebox.showerror("Error", "Invalid User credentials!")

    tk.Button(login_win, text="Login", command=check_login).pack(pady=10)


def show_admin_buttons():
    clear_buttons()

    tk.Button(main, text="Upload Dataset", command=upload_dataset, font=font1, bg="black", fg="white").place(x=1300, y=100)

    tk.Button(main, text="Preprocess Dataset", command=preprocess_data_tkinter, font=font1, bg="black", fg="white").place(x=1300, y=150)

    tk.Button(main, text="Train/Test Split", command=split_train_test_tkinter, font=font1, bg="black", fg="white").place(x=1300, y=200)

    tk.Button(main, text="Perform EDA", command=perform_eda_tkinter, font=font1, bg="black", fg="white").place(x=1300, y=250)

    tk.Button(main, text="Train LDA Model", command=lambda: train_lda_classifier_tkinter(X_train, y_train, X_test, y_test), font=font1, bg="black", fg="white").place(x=1300, y=300)
    tk.Button(main, text="Train SVC Model", command=lambda: train_kernel_svm_classifier_tkinter(X_train, y_train, X_test, y_test), font=font1, bg="black", fg="white").place(x=1300, y=350)
    tk.Button(main, text="Train QDA Model", command=lambda: train_qda_classifier_tkinter(X_train, y_train, X_test, y_test), font=font1, bg="black", fg="white").place(x=1300, y=400)

    tk.Button(main, text="Train Stacking Model", command=lambda: train_bagging_logistic_stacking_tkinter(X_train, y_train, X_test, y_test), font=font1, bg="black", fg="white").place(x=1300, y=450)

    tk.Button(main, text="Plot Model Performance", command=plot_model_performance_tkinter, font=font1, bg="black", fg="white").place(x=1300, y=500)


def show_user_buttons():

    clear_buttons()

    tk.Button(main, text="Upload Test Data", command=upload_test_data, font=font1, bg="white", fg="black").place(x=1300, y=300)
    tk.Button(main, text="Predict (Stacking Model)", command=predict_stacking_model_tkinter, font=font1, bg="white", fg="black").place(x=1300, y=350)
    tk.Button(main, text="View Model Performance Graph", command=plot_model_performance_tkinter, font=font1, bg="white", fg="black").place(x=1300, y=400)

    tk.Button(main, text="Exit", command=close, font=font1).place(x=1300, y=450)


def clear_buttons():
    for widget in main.winfo_children():
        if isinstance(widget, tk.Button) and widget not in [admin_button, user_button]:
            widget.destroy()

main = tk.Tk()
screen_width = main.winfo_screenwidth()
screen_height = main.winfo_screenheight()
main.geometry(f"{screen_width}x{screen_height}")

bg_image = Image.open("background.jpg")
bg_image = bg_image.resize((screen_width, screen_height), Image.LANCZOS) 
bg_photo = ImageTk.PhotoImage(bg_image)

bg_label = tk.Label(main, image=bg_photo)
bg_label.place(relwidth=1, relheight=1) 


def scroll_title(text, label, delay=200):
    def shift():
        nonlocal text
        text = text[1:] + text[0] 
        label.config(text=text)
        label.after(delay, shift) 
    shift()

font = ('times', 18, 'bold')
title = Label(main, text='Fertilizer Recommendation')
title.config(bg='pink', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

scroll_title('   Fertilizer Recommendation   ', title, delay=200)

font1 = ('times', 12, 'bold')
admin_button = tk.Button(main, text="ADMIN", command=lambda: authenticate("ADMIN"), font=font1, width=20, height=2, bg='LightBlue')
admin_button.place(x=35, y=620)

user_button = tk.Button(main, text="USER", command=lambda: authenticate("USER"), font=font1, width=20, height=2, bg='LightGreen')
user_button.place(x=35, y=720)


font1 = ('times', 12, 'bold')
text=Text(main,height=18,width=75,bg="black", fg="white", font=font1)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=35,y=220)
text.config(font=font1)
main.config(bg='Cyan2')
main.mainloop()
