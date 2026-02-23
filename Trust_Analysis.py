import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

# Load data
df = pd.read_csv("Survey_Result.csv")

# Rename columns to shorter names
df.columns = [
    'Timestamp', 'Age', 'Gender', 'Nationality', 'Familiarity',
    'Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7'
]

# Categorize Nationality 
def get_region(n):
    n = str(n).lower()
    arab_words = ['uae', 'emirati', 'emirates', 'emirate', 'arab', 'syria', 'sudanese']
    for w in arab_words:
        if w in n:
            return "Arab/Emirati"
    return "Western"

df['Region'] = df['Nationality'].apply(get_region)

# Likert scale conversion
scale = {
    'Agree': 5,
    'Somewhat agree': 4,
    'Somewhat disagree': 2,
    'Disagree': 1
}

trust_questions = ['Q1','Q2','Q3','Q4','Q5','Q6','Q7']

for q in trust_questions:
    df[q + '_num'] = df[q].map(scale)

# Average trust score
num_cols = [q + '_num' for q in trust_questions]
df['AvgTrust'] = df[num_cols].mean(axis=1)

# Group averages
west = df[df['Region'] == 'Western']
arab = df[df['Region'] == 'Arab/Emirati']

print(f"Sample sizes: Western={len(west)}, Arab/Emirati={len(arab)}")
print(f"Western avg trust: {west['AvgTrust'].mean():.2f}")
print(f"Arab/Emirati avg trust: {arab['AvgTrust'].mean():.2f}")

# -----------------------------
# 1. Average Trust Comparison
# -----------------------------
plt.figure(figsize=(6,4))
plt.bar(['Western', 'Arab/Emirati'], 
        [west['AvgTrust'].mean(), arab['AvgTrust'].mean()],
        color=['blue','green'])
plt.ylabel("Average Trust Score")
plt.title("Overall Trust in AI by Region")
plt.ylim(0,5)
plt.tight_layout()
plt.savefig('chart1_overall.png', dpi=300, bbox_inches='tight')
print("✓ Chart 1 saved: chart1_overall.png")

# -----------------------------
# 2. Trust Questions 
# -----------------------------
west_means = [west[q + '_num'].mean() for q in trust_questions]
arab_means = [arab[q + '_num'].mean() for q in trust_questions]

x = np.arange(len(trust_questions))
width = 0.35

plt.figure(figsize=(10,5))
plt.bar(x - width/2, west_means, width, label='Western', color='blue')
plt.bar(x + width/2, arab_means, width, label='Arab/Emirati', color='green')
plt.xticks(x, trust_questions)
plt.ylabel("Average Score")
plt.title("Trust Scores by Question")
plt.legend()
plt.ylim(0,5)
plt.tight_layout()
plt.savefig('chart2_questions.png', dpi=300, bbox_inches='tight')
print("✓ Chart 2 saved: chart2_questions.png")

# -----------------------------
# 3. Age Distribution Histogram
# -----------------------------
plt.figure(figsize=(8,5))
plt.hist([west['Age'], arab['Age']], bins=10,
         label=['Western', 'Arab/Emirati'], color=['blue','green'], alpha=0.7)
plt.xlabel("Age")
plt.ylabel("Number of Respondents")
plt.title("Age Distribution by Region")
plt.legend()
plt.tight_layout()
plt.savefig('chart3_age.png', dpi=300, bbox_inches='tight')
print("✓ Chart 3 saved: chart3_age.png")
# Calculate question differences
# -----------------------------
print("\n" + "="*60)
print("TRUST SCORE DIFFERENCES BY QUESTION:")
print("="*60)

question_names = {
    'Q1': 'Customer Service',
    'Q2': 'Reliable Information',
    'Q3': 'Treat Without Bias',
    'Q4': 'Explains Reasoning',
    'Q5': 'Personal Data',
    'Q6': 'Ethical Decisions',
    'Q7': 'Competitive Advantage'
}

differences = {}
for q in trust_questions:
    west_score = west[q + '_num'].mean()
    arab_score = arab[q + '_num'].mean()
    diff = arab_score - west_score
    differences[q] = diff
    print(f"{question_names[q]:<25} | West: {west_score:.2f} | Arab: {arab_score:.2f} | Diff: {diff:.2f}")

# Find the biggest difference
biggest = max(differences, key=lambda k: abs(differences[k]))
print(f"\nBiggest difference: {question_names[biggest]} ({differences[biggest]:.2f} points)")
print("="*60)

# -----------------------------
# Mann-Whitney U Test
# -----------------------------
u_stat, p_val = mannwhitneyu(west['AvgTrust'], arab['AvgTrust'], 
                              alternative='two-sided')

print("\n" + "="*50)
print("Mann-Whitney U Test Results:")
print("="*50)
print(f"U-statistic: {u_stat:.1f}")
print(f"P-value: {p_val:.6f}")

if p_val < 0.001:
    print("Result: HIGHLY SIGNIFICANT (p < 0.001)")
    print("→ The difference is extremely unlikely due to chance")
elif p_val < 0.05:
    print("Result: STATISTICALLY SIGNIFICANT (p < 0.05)")
    print("→ The difference is unlikely due to chance")
else:
    print("Result: NOT SIGNIFICANT (p ≥ 0.05)")
    print("→ Could be due to random variation")

print("="*50)
print("\nAnalysis complete! 3 charts saved.")
