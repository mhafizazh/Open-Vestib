import pickle
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_model(model_path):
    """
    Loads a model from a .pkl file.

    Args:
        model_path (str): The path to the .pkl file.

    Returns:
        The loaded model object.
    """
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

def estimate_bppv_likelihood(csv_path, window_size=150):
    # Load and extract features
    df = pd.read_csv(csv_path)
    features = df[['x_position', 'y_position', 'velocity', 'ellipse_size']].values

    # Handle invalid entries
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # Windowing
    num_windows = len(features) // window_size
    X_windowed = features[:num_windows * window_size].reshape(num_windows, -1)

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_windowed)

    # Predict probability of class 1 (BPPV)
    clf = joblib.load(model_path)
    y_proba = clf.predict_proba(X_scaled)[:, 1]

    # Assign probability back to full dataframe
    predict_scene['ml_prediction_proba'] = 0.0
    for i, proba in enumerate(y_proba):
        start = i * window_size
        end = start + window_size
        predict_scene.loc[start:end, 'ml_prediction_proba'] = proba
    
    # Segment into windows
    df['window_id'] = df.index // window_size

    # Compute average probability per segment
    segment_scores = df.groupby('window_id')['ml_prediction_proba'].mean().reset_index(name='segment_score')

    # Final likelihood is average segment confidence
    likelihood = segment_scores['segment_score'].mean() * 100

    print(segment_scores)
    print(f"\n🧠 Estimated likelihood of BPPV (based on ML confidence): {likelihood:.2f}%")
    # return likelihood
