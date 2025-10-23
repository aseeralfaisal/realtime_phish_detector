# RL Realtime Phish Detection

A reinforcement learning-based system for real-time phishing detection using Deep Q-Networks (DQN).

## Features
- Custom Gym environment (`PhishEnv`) for phishing detection.
- Supports URL, HTML, and DOM-based feature modes.
- Preprocessing and evaluation scripts included.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/A2Fais/rl_realtime_phish_detect.git
   cd rl_realtime_phish_detect
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
- Train the model: `python src/agent.py`
- Evaluate the model: `python analysis/evaluate_model.py`
- Test the model: `python analysis/test_model.py`
- Extract features: `python extractor/phiusiil_extractor.py`

## Project Structure
- `src/`: Core logic and environment.
- `data/`: Datasets and preprocessing scripts.
- `analysis/`: Evaluation and testing scripts.
- `extractor/`: Feature extraction tools.

