# Pro-Portion v1.1 - AI-Powered Facial Proportion Analysis Tool

##  Overview
Pro-Portion is an advanced facial proportion analysis system combining computer vision (MediaPipe) and machine learning (SVR) to provide step-by-step Loomis grid tutorials and proportion analysis for artists, photographers, and designers.

##  Features
- **Step-by-Step Loomis Grid Tutorial** - 6 progressive construction steps
- **Real-time Measurements** - Pixel-accurate facial measurements
- **ML Proportion Analysis** - Compare detected proportions vs classical ideals
- **Face Shape Classification** - Oval, Round/Square, Oblong categorization
- **Multi-face Detection** - Analyze group photos
- **REST API** - Easy integration with web/mobile apps

##  Quick Start

### Installation
```bash
# Clone repository
git clone <your-repo-url>
cd pro-portion

# Install dependencies
pip install -r requirements.txt

# Run server
uvicorn main:app --reload --port 8000
