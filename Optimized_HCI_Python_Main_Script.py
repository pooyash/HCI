#-------------------------------------------------------------------------------
# Name:        Optimized_VHCI_Python_Main_Script
# Purpose:     Personal research
# Author:      Pouya
# Created:     09/10/2024
# Copyright:   (c) Pouya 2024
# Licence:     <your licence>
#-------------------------------------------------------------------------------
#Libraries loaded
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Initialize variables

# Number of samples of DataSet
num_samples = 1000

# Generate Interaction_ID
interaction_ids = np.arange(1, num_samples + 1)

# Generate User_ID
user_ids = np.random.randint(1, 101, size=num_samples)  # 100 hypothetical users

# Generate Session_ID
session_ids = np.random.randint(1, 201, size=num_samples)  # 200 hypothetical meetings

# Generate Interaction Type
interaction_types = np.random.choice(['Click', 'Scroll', 'Type'], size=num_samples)

# Generate Interaction Duration with optimization (increased duration)
interaction_durations = np.random.uniform(1.0, 15.0, size=num_samples)  # Between 1 and 15 seconds

# Generate Emotional Responses with modified distribution
emotional_responses = np.random.choice(['Happy', 'Sad', 'Neutral'], p=[0.5, 0.3, 0.2], size=num_samples)

# Generate Satisfaction Score with optimization (increased average)
satisfaction_scores = np.random.randint(5, 11, size=num_samples)  # From 5 to 10

# Generate Error Count
error_counts = np.random.poisson(1, size=num_samples)  # Number of errors with Poisson distribution

# Generate Timestamp
start_date = datetime.now() - timedelta(days=1)  # Starting a day ago
timestamps = [start_date + timedelta(seconds=np.random.randint(0, 86400)) for _ in range(num_samples)]  # 86400 seconds in a day

# Combine all data into a DataFrame
data = {
    'Interaction_ID': interaction_ids,
    'User_ID': user_ids,
    'Session_ID': session_ids,
    'Interaction_Type': interaction_types,
    'Interaction_Duration': interaction_durations,
    'Emotional_Response': emotional_responses,
    'Satisfaction_Score': satisfaction_scores,
    'Error_Count': error_counts,
    'Timestamp': timestamps
}

df = pd.DataFrame(data)

# Save the dataset to a CSV file
df.to_csv('optimized_virtual_human_computer_interaction_dataset.csv', index=False)

print("Optimized dataset generated and saved as 'optimized_virtual_human_computer_interaction_dataset.csv'.")

# Plotting graphs

# 1. Optimized Distribution of Interaction Types Plot
plt.figure(num="Figure - Optimized Distribution of Interaction Types Plot", figsize=(8, 6))
sns.countplot(data=df, x='Interaction_Type', palette='viridis')
plt.title('Optimized Distribution of Interaction Types', fontsize=16)
plt.xlabel('Interaction Type', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
mng = plt.get_current_fig_manager()
# mng.resize(*mng.window.maxsize())
mng.window.state('zoomed') #works fine on Windows!
# Saving the figure.
plt.savefig("Figure - Optimized Distribution of Interaction Types Plot.jpg")
plt.show()

# 2. Optimized Distribution of Emotional Responses Plot
plt.figure(num="Figure - Optimized Distribution of Emotional Responses Plot", figsize=(8, 6))
sns.countplot(data=df, x='Emotional_Response', palette='coolwarm')
plt.title('Optimized Distribution of Emotional Responses', fontsize=16)
plt.xlabel('Emotional Response', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
mng = plt.get_current_fig_manager()
# mng.resize(*mng.window.maxsize())
mng.window.state('zoomed') #works fine on Windows!
# Saving the figure.
plt.savefig("Figure - Optimized Distribution of Emotional Responses Plot.jpg")
plt.show()

# 3. Optimized Satisfaction Score by Interaction Type Plot
plt.figure(num="Figure - Optimized Satisfaction Score by Interaction Type Plot", figsize=(8, 6))
sns.boxplot(data=df, x='Interaction_Type', y='Satisfaction_Score', palette='pastel')
plt.title('Optimized Satisfaction Score by Interaction Type', fontsize=16)
plt.xlabel('Interaction Type', fontsize=14)
plt.ylabel('Satisfaction Score', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
mng = plt.get_current_fig_manager()
# mng.resize(*mng.window.maxsize())
mng.window.state('zoomed') #works fine on Windows!
# Saving the figure.
plt.savefig("Figure - Optimized Satisfaction Score by Interaction Type Plot.jpg")
plt.show()

# 4. Optimized Error Count by Interaction Type Plot
plt.figure(num="Figure - Optimized Error Count by Interaction Type Plot", figsize=(8, 6))
sns.boxplot(data=df, x='Interaction_Type', y='Error_Count', palette='magma')
plt.title('Optimized Error Count by Interaction Type', fontsize=16)
plt.xlabel('Interaction Type', fontsize=14)
plt.ylabel('Error Count', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
mng = plt.get_current_fig_manager()
# mng.resize(*mng.window.maxsize())
mng.window.state('zoomed') #works fine on Windows!
# Saving the figure.
plt.savefig("Figure - Optimized Error Count by Interaction Type Plot.jpg")
plt.show()

# 5. Optimized Interaction Duration vs Satisfaction Score Plot
plt.figure(num="Figure - Optimized Interaction Duration vs Satisfaction Score Plot", figsize=(8, 6))
sns.scatterplot(data=df, x='Interaction_Duration', y='Satisfaction_Score', hue='Emotional_Response', palette='dark')
plt.title('Optimized Interaction Duration vs Satisfaction Score', fontsize=16)
plt.xlabel('Interaction Duration (seconds)', fontsize=14)
plt.ylabel('Satisfaction Score', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(title='Emotional Response', fontsize=12)
mng = plt.get_current_fig_manager()
# mng.resize(*mng.window.maxsize())
mng.window.state('zoomed') #works fine on Windows!
# Saving the figure.
plt.savefig("Figure - Optimized Interaction Duration vs Satisfaction Score Plot.jpg")
plt.show()

def main():
    pass
if __name__ == '__main__':
    main()
