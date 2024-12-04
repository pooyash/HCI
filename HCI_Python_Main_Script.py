#-------------------------------------------------------------------------------
# Name:        HCI_Python_Main_Script
# Purpose:     Personal research
# Author:      Pouya
# Created:     09/10/2024
# Copyright:   (c) Pouya 2024
# Licence:     <your licence>
#-------------------------------------------------------------------------------
#Libraries loaded
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Initial settings
num_samples = 1000

# Variables defining
ages = [random.randint(18, 65) for _ in range(num_samples)]
genders = ['Male', 'Female']
genders = [random.choice(genders) for _ in range(num_samples)]
experience_levels = ['Low', 'Medium', 'High']
experience_levels = [random.choice(experience_levels) for _ in range(num_samples)]
emotional_states = ['Happy', 'Nervous', 'Anxious']
emotional_states = [random.choice(emotional_states) for _ in range(num_samples)]
environment_types = ['Simple', 'Complex', 'Scary']
environment_types = [random.choice(environment_types) for _ in range(num_samples)]
object_interactions = [random.randint(1, 10) for _ in range(num_samples)]
interactivity_levels = ['Low', 'Medium', 'High']
interactivity_levels = [random.choice(interactivity_levels) for _ in range(num_samples)]
response_times = [round(random.uniform(1.0, 5.0), 1) for _ in range(num_samples)]
response_accuracies = ['Low', 'Medium', 'High']
response_accuracies = [random.choice(response_accuracies) for _ in range(num_samples)]
user_satisfaction = [random.randint(1, 5) for _ in range(num_samples)]

# DataSet creation
data = {
    "Age": ages,
    "Gender": genders,
    "Experience_Level": experience_levels,
    "Emotional_State": emotional_states,
    "Environment_Type": environment_types,
    "Object_Interaction": object_interactions,
    "Interactivity_Level": interactivity_levels,
    "Response_Time": response_times,
    "Response_Accuracy": response_accuracies,
    "User_Satisfaction": user_satisfaction
}

# DataSet to dataframe converting
df = pd.DataFrame(data)

# DataSet printing
print(df.head())

# Categorical variables to numerical envoding
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
df['Experience_Level'] = df['Experience_Level'].map({'Low': 0, 'Medium': 1, 'High': 2})
df['Emotional_State'] = df['Emotional_State'].map({'Happy': 0, 'Nervous': 1, 'Anxious': 2})
df['Environment_Type'] = df['Environment_Type'].map({'Simple': 0, 'Complex': 1, 'Scary': 2})
df['Interactivity_Level'] = df['Interactivity_Level'].map({'Low': 0, 'Medium': 1, 'High': 2})
df['Response_Accuracy'] = df['Response_Accuracy'].map({'Low': 0, 'Medium': 1, 'High': 2})

# Plot histogram of Age
plt.figure(num="Figure - Age Distribution Plot", figsize=(8, 6))
sns.histplot(df['Age'], bins=15, kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
mng = plt.get_current_fig_manager()
# mng.resize(*mng.window.maxsize())
mng.window.state('zoomed') #works fine on Windows!
# Saving the figure.
plt.savefig("Figure - Age Distribution Plot.jpg")
plt.show()

# Box plot of User Satisfaction
plt.figure(num="Figure - User Satisfaction Plot", figsize=(8, 6))
sns.boxplot(x='User_Satisfaction', data=df)
plt.title('User Satisfaction Box Plot')
plt.xlabel('User Satisfaction')
mng = plt.get_current_fig_manager()
# mng.resize(*mng.window.maxsize())
mng.window.state('zoomed') #works fine on Windows!
# Saving the figure.
plt.savefig("Figure - User Satisfaction Plot.jpg")
plt.show()

# Plot of Correlation Heatmap of variables
plt.figure(num="Figure - Correlation Heatmap Plot", figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
mng = plt.get_current_fig_manager()
# mng.resize(*mng.window.maxsize())
mng.window.state('zoomed') #works fine on Windows!
# Saving the figure.
plt.savefig("Figure - Correlation Heatmap Plot.jpg")
plt.show()

def main():
    pass
if __name__ == '__main__':
    main()