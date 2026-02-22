"""Simple keyword-based category classifier as fallback."""

def classify_complaint_simple(text: str) -> tuple[str, float]:
    """
    Simple keyword-based classification for complaints.
    Returns (category, confidence).
    """
    text_lower = text.lower()
    
    # Keywords for each category
    sanitation_keywords = [
        'garbage', 'kachra', 'कचरा', 'कूड़ा', 'dustbin', 'waste', 'smell', 'badbu', 'बदबू',
        'drain', 'nali', 'नाली', 'dirty', 'ganda', 'गंदा', 'overflow', 'sanitation'
    ]
    
    water_keywords = [
        'water', 'pani', 'पानी', 'supply', 'leakage', 'pipeline', 'pressure',
        'tank', 'tanki', 'टंकी', 'leak', 'waste'
    ]
    
    transport_keywords = [
        'road', 'sadak', 'सड़क', 'pothole', 'streetlight', 'bus', 'traffic',
        'accident', 'durghatna', 'दुर्घटना', 'shelter', 'गड्ढे', 'potholes'
    ]
    
    # Count matches
    sanitation_score = sum(1 for kw in sanitation_keywords if kw in text_lower)
    water_score = sum(1 for kw in water_keywords if kw in text_lower)
    transport_score = sum(1 for kw in transport_keywords if kw in text_lower)
    
    # Determine category
    scores = {
        'Sanitation': sanitation_score,
        'Water Supply': water_score,
        'Transportation': transport_score
    }
    
    category = max(scores, key=scores.get)
    max_score = scores[category]
    total = sum(scores.values())
    
    # Calculate confidence
    if total == 0:
        confidence = 0.33  # Equal probability if no keywords found
    else:
        confidence = max_score / total
    
    # Ensure minimum confidence
    confidence = max(0.5, confidence)
    
    return category, confidence
