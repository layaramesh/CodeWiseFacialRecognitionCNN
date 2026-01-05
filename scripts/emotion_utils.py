"""
Shared utility functions for emotion classification analysis
"""
import re

def extract_classification_from_filename(filename):
    """Extract emotion classification from filename like 'angry_001.jpg' or 'angry001.jpg'"""
    match = re.match(r'([a-z]+)_?', filename.lower())
    if match:
        emotion = match.group(1)
        # Map filename prefixes to standard emotion names
        emotion_map = {
            'angry': 'Anger',
            'sad': 'Sadness',
            'happy': 'Happiness',
            'fear': 'Fear',
            'disgust': 'Disgust',
            'neutral': 'Neutral',
            'surprise': 'Surprise'
        }
        return emotion_map.get(emotion, emotion.capitalize())
    return ''

def extract_emotion(cell_value):
    """Extract emotion name from cell value like 'Anger (91.26%)'"""
    if not cell_value or cell_value.strip() == '':
        return None
    match = re.match(r'([A-Za-z]+)', cell_value.strip())
    if match:
        emotion = match.group(1)
        # Normalize emotion names
        emotion_map = {
            'anger': 'Anger',
            'sadness': 'Sadness',
            'happiness': 'Happiness',
            'fear': 'Fear',
            'disgust': 'Disgust',
            'neutral': 'Neutral',
            'surprise': 'Surprise'
        }
        return emotion_map.get(emotion.lower(), emotion.capitalize())
    return None

def get_label(emotion):
    """Categorize emotion into 'Needs Help' or 'Understands Material'"""
    needs_help_emotions = ['Fear', 'Anger', 'Surprise', 'Disgust', 'Sadness']
    
    if emotion in needs_help_emotions:
        return 'Needs Help'
    else:
        return 'Understands Material'

def get_hybrid_prediction(classification, cnn_prediction, ferplus_prediction):
    """
    Get prediction based on classification: 
    use CNN for needs-help emotions, ferplus for understand-material emotions
    """
    needs_help_emotions = ['Fear', 'Anger', 'Surprise', 'Disgust', 'Sadness']
    
    if classification in needs_help_emotions:
        return cnn_prediction if cnn_prediction else ''
    else:
        return ferplus_prediction if ferplus_prediction else ''

def calculate_metrics(tp, tn, fp, fn):
    """Calculate Accuracy, Precision, Recall, and F1 Score"""
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1_score
    }

# Standard emotion labels
EMOTION_LABELS = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']
EMOTION_LABELS_SHORT = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
LABEL_CATEGORIES = ['Needs Help', 'Understands Material']
