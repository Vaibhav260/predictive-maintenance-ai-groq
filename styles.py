
def get_custom_css():
    """Returns the custom CSS as a string"""
    return """
    <style>
        /*  MAIN TITLE STYLING */
        .main-title {
            font-size: 3rem;
            font-weight: 700;
            text-align: center;
            background: linear-gradient(120deg, #1f77b4, #2ca02c, #ff7f0e);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
            padding: 1rem 0;
            animation: fadeInDown 0.8s ease-out;
        }
        
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 2rem;
            font-size: 1.15rem;
            font-weight: 400;
        }
        
        /*  BUTTON STYLING */
        .stButton > button {
            width: 100%;
            background: linear-gradient(90deg, #1f77b4, #2ca02c);
            color: white;
            border: none;
            padding: 0.85rem 1.5rem;
            font-size: 1.05rem;
            font-weight: 700;
            border-radius: 10px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
            cursor: pointer;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .stButton > button:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 16px rgba(0,0,0,0.25);
            background: linear-gradient(90deg, #145a8c, #1e7d1e);
        }
        
        .stButton > button:active {
            transform: translateY(-1px);
        }
        
        .stButton > button:disabled {
            background: linear-gradient(90deg, #cccccc, #999999);
            cursor: not-allowed;
            transform: none;
            opacity: 0.6;
        }
        
        /*   BEAUTIFUL RESULT BOXES */
        .result-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem 1.5rem;
            border-radius: 15px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.2);
            color: white;
            margin: 1.5rem 0;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            text-align: center;
        }
        
        .result-box:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 35px rgba(0,0,0,0.3);
        }
        
        .result-box h2 {
            margin: 0;
            font-size: 2.5rem;
            font-weight: 700;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }
        
        .result-box p {
            margin: 0.8rem 0 0 0;
            font-size: 1.1rem;
            opacity: 0.95;
            font-weight: 500;
        }
        
        /* Different colored boxes */
        .result-box-success {
            background: linear-gradient(135deg, #2ca02c 0%, #1e7d1e 100%);
        }
        
        .result-box-warning {
            background: linear-gradient(135deg, #ff7f0e 0%, #d68910 100%);
        }
        
        .result-box-danger {
            background: linear-gradient(135deg, #d62728 0%, #8b0000 100%);
        }
        
        .result-box-info {
            background: linear-gradient(135deg, #1f77b4 0%, #145a8c 100%);
        }
        
        .result-box-purple {
            background: linear-gradient(135deg, #9b59b6 0%, #8e44ad 100%);
        }
        
        .result-box-pink {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }
        
        /* SECTION HEADERS  */
        .section-header {
            font-size: 1.8rem;
            font-weight: 700;
            color: #1f77b4;
            margin: 2rem 0 1.5rem 0;
            padding-bottom: 0.8rem;
            border-bottom: 3px solid #1f77b4;
        }
        
        /*  STATUS BADGES */
        .status-badge {
            display: inline-block;
            padding: 0.7rem 1.5rem;
            border-radius: 25px;
            font-weight: 700;
            font-size: 1.1rem;
            margin: 1rem 0;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        
        .badge-success {
            background: linear-gradient(135deg, #d4edda, #c3e6cb);
            color: #155724;
            border: 2px solid #28a745;
        }
        
        .badge-warning {
            background: linear-gradient(135deg, #fff3cd, #ffeaa7);
            color: #856404;
            border: 2px solid #ffc107;
        }
        
        .badge-danger {
            background: linear-gradient(135deg, #f8d7da, #f5c6cb);
            color: #721c24;
            border: 2px solid #dc3545;
        }
        
        /*  REPORT BOX */
        .report-box {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-left: 5px solid #1f77b4;
            padding: 1.8rem;
            border-radius: 10px;
            margin: 1.5rem 0;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            font-size: 1.05rem;
            line-height: 1.8;
            color: #333;
        }
        
        /*  INPUT STYLING */
        .stNumberInput > div > div > input,
        .stTextArea textarea {
            border-radius: 8px;
            border: 2px solid #e0e0e0;
            padding: 0.75rem;
            font-size: 1rem;
            transition: all 0.3s ease;
        }
        
        .stNumberInput > div > div > input:focus,
        .stTextArea textarea:focus {
            border-color: #1f77b4;
            box-shadow: 0 0 0 3px rgba(31, 119, 180, 0.15);
        }
        
      
        .stTabs [data-baseweb="tab-list"] {
            gap: 1rem;
            
            padding: 0.5rem;
            border-radius: 10px;
        }
        
        .stTabs [data-baseweb="tab"] {
            padding: 1rem 2rem;
            font-size: 1.1rem;
            font-weight: 600;
            border-radius: 8px;
        }
        
        /* ========== REMOVE EXTRA PADDING ========== */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        
        /* ========== ANIMATIONS ========== */
        @keyframes fadeInDown {
            from {
                opacity: 0;
                transform: translateY(-30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        /* ========== CHART SPACING ========== */
        div[data-testid="stVerticalBlock"] > div {
            margin-bottom: 1.5rem;
        }
        
        .stAltairChart {
            margin-bottom: 1.5rem !important;
        }
    </style>
    """
