import matplotlib.pyplot as plt

# Recall and precision values for system S1
recall_s1 = [0.04, 0.16, 0.2, 0.32, 0.48, 0.52, 0.64, 0.64, 0.76, 0.8]
precision_s1 = [0.1, 0.1, 0.3, 0.4, 0.5, 0.5, 0.57, 0.5, 0.56, 0.5]

# Recall and precision values for system S2
recall_s2 = [0.04, 0.16, 0.2, 0.2, 0.2, 0.2, 0.2, 0.32, 0.32, 0.36]
precision_s2 = [0.1, 0.2, 0.3, 0.25, 0.2, 0.16, 0.14, 0.25, 0.22, 0.2]

# Plotting the exact recall/precision graph for each system
plt.figure(figsize=(10, 6))
plt.plot(recall_s1, precision_s1, marker='o', label='System S1')
plt.plot(recall_s2, precision_s2, marker='o', label='System S2')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Exact Recall/Precision Graph')
plt.legend()
plt.grid(True)
plt.show()

# Function to calculate F1 measure
def calculate_f1(precision, recall):
    f1 = []
    for i in range(len(precision)):
        if precision[i] + recall[i] == 0:
            f1.append(0)
        else:
            f1.append(2 * (precision[i] * recall[i]) / (precision[i] + recall[i]))
    return f1

# Calculate F1 measure for each system
f1_s1 = calculate_f1(precision_s1, recall_s1)
f1_s2 = calculate_f1(precision_s2, recall_s2)

# Plotting the F1 measure for both systems
plt.figure(figsize=(10, 6))
plt.plot(recall_s1, f1_s1, marker='o', label='System S1')
plt.plot(recall_s2, f1_s2, marker='o', label='System S2')
plt.xlabel('Recall')
plt.ylabel('F1 Measure')
plt.title('F1 Measure Comparison')
plt.legend()
plt.grid(True)
plt.show()
