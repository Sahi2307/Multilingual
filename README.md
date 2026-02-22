# 🏛️ Explainable Multilingual Civic Complaint Resolution System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive AI-powered civic complaint management system with multilingual support (English, Hindi, Hinglish) and explainable AI using SHAP. Built with MuRIL and XGBoost for intelligent complaint routing and prioritization.

## 🎯 Key Features

### 🌐 Multilingual Support
- Accept complaints in **English**, **Hindi**, and **Hinglish**
- Automatic language detection and processing
- MuRIL-based understanding of code-mixed text

### 🤖 AI-Powered Intelligence
- **Category Classification**: MuRIL transformer (≥94% accuracy)
- **Urgency Prediction**: XGBoost classifier (≥89% accuracy)
- **SHAP Explanations**: Simple, natural language sentences for all AI decisions
- **Smart Routing**: Automatic department assignment
- **Priority Scoring**: Queue position based on urgency and severity

### 👥 Role-Based Access Control

#### Citizens
- File complaints with location data and photos
- Track status via visual timeline (Registered → Assigned → In Progress → Completed)
- View complaint history and AI explanations

#### Officials
- Department-specific dashboard
- View unassigned complaints in department queue
- Update status with remarks
- Real-time metrics (pending, assigned, resolved)

#### Administrators
- Full system oversight and analytics
- **Department Management**: Create and manage departments
- User management (approval/suspension)
- Global complaint assignment
- System settings and password management

### 🔒 Security Features
- Secure password hashing (bcrypt)
- Session management with expiration
- Role-based page access control
- Mandatory password change on first login
- CAPTCHA after failed login attempts

## 📊 Project Architecture

The system follows a 6-module architecture:

### MODULE 1: Data Preparation Module
- **File**: `src/data_preparation.py`
- Generates synthetic multilingual complaints
- Creates balanced dataset with 500 samples.
- Computes severity scores and emergency keywords

### MODULE 2: Feature Extraction & Explainability Module
- **Files**: `src/feature_extraction.py`, `src/explainability.py`
- **Features**: 776-dimensional (768 MuRIL + 8 structured)
- **Explainability**: SHAP with integrated token-level importance for categories and factor-wise importance for urgency.

### MODULE 3: Model Training Module
- **MuRIL**: `src/train_category_model.py` - Fine-tunes Google's MuRIL for category classification
- **XGBoost**: `src/train_urgency_model.py` - Trains gradient boosting model for urgency prediction

### MODULE 4: Complaint Processing Module
- **Backend**: `src/complaint_processor.py`
- End-to-end processing pipeline (Language detection -> Category -> Urgency -> Department -> ETA)

### MODULE 5: User Interface Module
- **Frontend**: Streamlit-based UI (`Home.py` + `pages/`)
- Interactive dashboard for Citizens, Officials, and Admins
- Visual timeline, SHAP keyword bar charts, and factor importance graphs.

### MODULE 6: Smart Clustering & Dynamic Escalation
- **Logic**: Integrated within `src/complaint_processor.py` and `utils/database.py`
- **Geospatial Clustering**: Groups complaints within a **2km radius** using Haversine distance.
- **Dynamic Priority**: 
    - 📁 **3+ reports**: Elevates priority to **High**.
    - 🚨 **5+ reports**: Forces priority to **Critical** (First Priority).
    - ⚡ **Burst Detection**: Escalates to Critical if **5+ reports** occur within **1 hour**.
- **Admin Consolidation**: Provides a unified view of clusters to prevent redundancy.

## 🚀 Quick Start

### Prerequisites
- Python 3.8 to 3.11 (Note: Python 3.12+ might have compatibility issues with some ML libs)
- 4GB+ RAM
- CUDA-capable GPU (optional, for faster training)

### Installation

1. **Clone Repository**
```bash
git clone https://github.com/Sahi2307/multilingual.git
cd multilingual
```

2. **Create Virtual Environment**
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

> **⚠️ Installation Note**: If you encounter errors related to `numpy`, ensure you are using a version `< 2.0.0` (as specified in requirements.txt). You can fix this with:
> `pip install "numpy<2.0.0"`

4. **Initialize Database**
```bash
python -c "from utils.database import init_db; import os; init_db(os.getcwd())"
```

This creates:
- SQLite database at `data/civic_complaints.db`
- Default admin account: `admin@civiccomplaints.gov`
- Default official accounts (Sanitation, Water, Transportation)

### Training the Models

**Step 1: Generate Training Data**
```bash
python src/data_preparation.py
python src/feature_extraction.py
```

**Step 2: Train Category Model (MuRIL)**
```bash
python src/train_category_model.py
```
- Training time: ~10-15 minutes (GPU) / ~30-45 minutes (CPU)
- Output: `models/muril_category_classifier/`
- **Note**: MuRIL model is not included in the repository due to size (~500MB). You must train it locally.

**Step 3: Train Urgency Model (XGBoost)**
```bash
python src/train_urgency_model.py
```
- Training time: ~2-3 minutes
- Output: `models/xgboost_urgency_predictor.pkl`

### Running the Application

```bash
streamlit run Home.py
```

Access at: **http://localhost:8501**

## 👤 Default Accounts

### Admin Account
- **Email**: `admin@civiccomplaints.gov`
- **Note**: Change password immediately after first login

### Official Accounts
| Department | Email | 
|------------|-------|
| Sanitation | `sanitation_official@civiccomplaints.gov` | 
| Water Supply | `watersupply_official@civiccomplaints.gov` | 
| Transportation | `transportation_official@civiccomplaints.gov` |

### Citizen Accounts
- Register via the Home page
- No approval required for citizens

## 📁 Project Structure

```
civic-complaint-system/
├── Home.py                      # Landing page with login/registration
├── pages/
│   ├── 2_File_Complaint.py      # Complaint submission with AI predictions
│   ├── 3_My_Complaints.py       # Citizen complaint history
│   ├── 4_Track_Complaint.py     # Real-time tracking with timeline
│   ├── 5_Official_Dashboard.py  # Official workflow management
│   └── 6_Admin_Panel.py         # System administration
├── src/
│   ├── data_preparation.py      # Phase 1: Synthetic data generation
│   ├── feature_extraction.py    # Phase 2: Feature engineering
│   ├── explainability.py        # Phase 2: SHAP explainers
│   ├── train_category_model.py  # Phase 3: MuRIL training
│   ├── train_urgency_model.py   # Phase 3: XGBoost training
│   ├── complaint_processor.py   # Phase 4: Production pipeline
│   └── independent_test.py      # Independent evaluation on unseen data
├── utils/
│   ├── database.py              # SQLite operations
│   ├── auth.py                  # Authentication & password management
│   ├── session_manager.py       # Session handling
│   ├── ui.py                    # UI components & styling
│   ├── helpers.py               # Utility functions
│   └── notifications.py         # Notification system
├── models/                      # Trained ML models
│   ├── muril_category_classifier/
│   ├── xgboost_urgency_predictor.pkl
│   └── feature_scaler.pkl
├── data/
│   ├── civic_complaints.db      # SQLite database
│   ├── civic_complaints.csv     # Training dataset (6K)
│   ├── complaints.csv           # Full complaint corpus (6K, 3 languages)
│   └── independent_test.csv     # Independent test set (auto-generated)
└── requirements.txt             # Python dependencies
```

## 🔧 Technical Stack

### Machine Learning
- **MuRIL** (google/muril-base-cased): Multilingual BERT for category classification
- **XGBoost**: Gradient boosting for urgency prediction
- **SHAP**: Model explainability and interpretability
- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face library

### Backend
- **Streamlit**: Web application framework
- **SQLite**: Embedded database
- **bcrypt**: Password hashing
- **Pandas/NumPy**: Data processing

### Frontend
- **Streamlit Components**: Interactive UI elements
- **Matplotlib**: Visualization
- **Custom CSS**: Styling and theming

## 📈 Model Performance

### Training & Validation (6,000 complaints from `civic_complaints.csv`)

The models were trained on Google Colab (T4 GPU) using the **6K `civic_complaints.csv`** dataset, split as follows:
- **Train**: 4,200 samples (70%) | **Val**: 900 samples (15%) | **Test**: 900 samples (15%)
- **Languages**: English, Hindi, Hinglish (2,000 each)
- **Features**: 776-dimensional (768 MuRIL embeddings + 8 structured)

#### Category Classification (XGBoost + MuRIL Embeddings)

| Metric | Test Set Score |
|--------|---------------|
| **Accuracy** | **97.44%** |
| **F1-Score** (weighted) | **0.97** |

| Category | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Sanitation | 0.98 | 0.98 | 0.98 | 371 |
| Water Supply | 0.96 | 0.93 | 0.95 | 176 |
| Transportation | 0.97 | 0.99 | 0.98 | 353 |

#### Urgency Prediction (XGBoost)

| Metric | Test Set Score |
|--------|---------------|
| **Accuracy** | **91.56%** |
| **F1-Score** (weighted) | **0.92** |

| Urgency Level | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| Critical | 0.96 | 0.95 | 0.96 | 392 |
| High | 0.86 | 0.84 | 0.85 | 178 |
| Medium | 0.88 | 0.90 | 0.89 | 161 |
| Low | 0.93 | 0.96 | 0.95 | 169 |


## 🎨 UI Features

### Visual Timeline
- Progress tracking with 4 stages
- Color-coded status indicators
- Real-time updates from officials

### SHAP Explanations
- Word-level importance for category prediction
- Feature importance for urgency prediction
- Natural language summaries

### Responsive Design
- Mobile-friendly interface
- Dark mode support
- Accessible color schemes

## 🔐 Security Best Practices

1. **Password Policy**: Minimum 8 characters, uppercase, lowercase, number, special character
2. **Session Management**: Automatic expiration after inactivity
3. **Role Verification**: Server-side access control on every page
4. **SQL Injection Prevention**: Parameterized queries
5. **XSS Protection**: Input sanitization

## 📝 Database Schema

### Core Tables
- **users**: User accounts with role-based access
- **complaints**: Complaint records with AI predictions
- **departments**: Department information
- **status_updates**: Complaint status history
- **model_predictions**: AI prediction logs
- **sessions**: User session management

## 🧪 Testing

Run explainability tests:
```bash
python src/test_explainability.py
```

Run independent model evaluation (on unseen data from `complaints.csv`):
```bash
python -m src.independent_test
```

Benchmark processing performance:
```bash
python src/benchmark_processing.py
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Google Research**: MuRIL multilingual model
- **Hugging Face**: Transformers library
- **Streamlit**: Web application framework
- **SHAP**: Explainability framework

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Built with ❤️ for better civic governance**
