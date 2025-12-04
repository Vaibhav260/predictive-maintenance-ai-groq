
def create_result_box(value, label, box_type="info"):
    """
    Create a beautiful colored result box
    
    Parameters:
    - value: The main value to display (e.g., "85.5%", "High Risk")
    - label: The label/description (e.g., "Failure Probability")
    - box_type: Type of box - "success", "warning", "danger", "info", "purple", "pink"
    
    Returns:
    - HTML string for the result box
    """
    box_class = f"result-box result-box-{box_type}"
    
    return f"""
    <div class="{box_class}">
        <h2>{value}</h2>
        <p>{label}</p>
    </div>
    """


def create_status_badge(status_text, badge_type="success"):
    """
    Create a status badge
    
    Parameters:
    - status_text: Text to display (e.g., "Healthy", "High Risk")
    - badge_type: "success", "warning", or "danger"
    
    Returns:
    - HTML string for the badge
    """
    return f"""
    <div class="status-badge badge-{badge_type}">
        {status_text}
    </div>
    """


def create_metric_cards(prob, pred):
    """
    Create a set of three beautiful metric cards based on prediction
    
    Parameters:
    - prob: Failure probability (0-1)
    - pred: Prediction (0 or 1)
    
    Returns:
    - Tuple of (card1_html, card2_html, card3_html)
    """
    # Determine box type based on probability
    if prob >= 0.7:
        prob_box_type = "danger"
    elif prob >= 0.5:
        prob_box_type = "warning"
    else:
        prob_box_type = "success"
    
    # Card 1: Probability
    card1 = create_result_box(
        value=f"{prob*100:.1f}%",
        label="Failure Probability",
        box_type=prob_box_type
    )
    
    # Card 2: Status
    status_text = "⚠️ High Risk" if pred == 1 else "✅ Healthy"
    card2 = create_result_box(
        value=status_text,
        label="Machine Status",
        box_type="purple"
    )
    
    # Card 3: Confidence
    confidence = prob if pred == 1 else (1 - prob)
    card3 = create_result_box(
        value=f"{confidence*100:.1f}%",
        label="Model Confidence",
        box_type="pink"
    )
    
    return card1, card2, card3


def create_report_box(text):
    """
    Create a styled report box for AI recommendations
    
    Parameters:
    - text: The report text to display
    
    Returns:
    - HTML string for the report box
    """
    return f"""
    <div class="report-box">
        {text}
    </div>
    """
