Project Overview
This project focuses on improving the loan approval process using Supervised Deep Learning and Offline Reinforcement Learning (RL).
The goal was to build predictive and decision-making systems that help a fintech company decide whether to approve or deny a loan to maximize long-term financial returns.
The project follows the complete ML pipeline:
✔ Exploratory Data Analysis
✔ Feature Engineering
✔ Deep Learning Classification
✔ Offline RL Policy Optimization
✔ Comparative Analysis & Business Interpretation


1. Exploratory Data Analysis (EDA) & Preprocessing
I used the LendingClub accepted_2007_to_2018.csv dataset and performed detailed exploratory analysis to understand borrower profiles and factors affecting loan defaults.
Key steps performed:
✔ Data Cleaning
Filtered the dataset to only "Fully Paid" and "Charged Off" loans.
Removed irrelevant or highly sparse columns.
Handled missing values using median/most-frequent strategies.
Converted interest rate (% string) into numeric.
Encoded categorical variables (Label Encoding / One-Hot).
✔ Feature Engineering
I selected a meaningful subset of predictive features such as:
Loan characteristics: loan_amnt, int_rate, term
Financial background: annual_inc, dti, revol_util
Credit factors: delinq_2yrs, total_acc
Categorical risk indicators: grade, home_ownership, purpose


2. Deep Learning Model – Default Prediction
I built and trained a Multi-Layer Perceptron (MLP) to predict the probability that a borrower will default.
Model Architecture
Input layer → Dense(128) → ReLU → Dropout
Dense(64) → ReLU → Dropout
Output layer → Sigmoid (binary probability)
Training Details
Loss: Binary Cross-Entropy
Optimizer: Adam
Metrics: ROC-AUC, F1-Score
What I Achieved
The model learned risk patterns from borrower features.
Output probabilities can be converted into a loan approval policy by using a threshold (e.g., reject if default probability > 0.5).
This forms the baseline supervised-learning policy.


3. Offline Reinforcement Learning Agent
Next, I reframed the loan approval problem as an offline RL decision-making task.
RL Environment Setup
State (s): Applicant feature vector
Actions (a):
0 = Deny Loan
1 = Approve Loan
Reward (r):
Approve + Fully Paid → + loan_amnt * int_rate
Approve + Default → - loan_amnt
Deny → 0 (no gain, no loss)
Algorithm Used
I used a modern offline RL algorithm (e.g., CQL, IQL, or BCQ) via the d3rlpy library.
Training Output
The agent learned a profit-maximizing policy, not just a risk-minimizing one.
It sometimes approves loans that the classifier would reject—because high interest returns outweigh risk.
Policy Evaluation
I computed the Estimated Policy Value, representing expected future returns of the policy on historical data.


4. Model Comparison & Analysis
This is the core analytical insight of the project.
Supervised Model (MLP)
Optimized for classification (AUC / F1)
Learns to predict risk
Conservative policy (tends to reduce defaults)
Offline RL Model
Optimized for profit
Learns to balance risk vs reward
May approve some high-risk but high-profit loans
