import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic class data
np.random.seed(42)
class_1 = np.random.multivariate_normal(mean=[2, 2], cov=[[1, 0.5], [0.5, 1]], size=50)
class_2 = np.random.multivariate_normal(mean=[6, 6], cov=[[1, -0.3], [-0.3, 1]], size=50)

# Compute class means
mu_1 = np.mean(class_1, axis=0)
mu_2 = np.mean(class_2, axis=0)

# Define the original projection direction (e.g., diagonal)
v = np.array([1, 1.5]) / np.sqrt(2)  # 45-degree normalized direction

# Rotate v by θ 
theta_deg = 80
theta = np.radians(theta_deg)  # Convert to radians

rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
v_new = rotation_matrix @ v  # Rotate the vector

# Compute within-class scatter matrix Sw
S1 = np.sum([(x - mu_1).reshape(-1, 1) @ (x - mu_1).reshape(1, -1) for x in class_1], axis=0)
S2 = np.sum([(x - mu_2).reshape(-1, 1) @ (x - mu_2).reshape(1, -1) for x in class_2], axis=0)
S_W = S1 + S2

# Compute between-class scatter matrix Sb
mu_diff = (mu_2 - mu_1).reshape(-1, 1)
S_B = mu_diff @ mu_diff.T

# Solve the generalized eigenvalue problem for Sw^-1 Sb
eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(S_W) @ S_B)

# Select the eigenvector corresponding to the largest eigenvalue (optimal direction)
v_best = eigenvectors[:, np.argmax(eigenvalues)]

# Define a function to plot projection lines
def plot_projection_line(v, color, label):
    # Plots a projection line along direction v.
    # Define a center point for reference (e.g., the midpoint of the means)
    center = (mu_1 + mu_2) / 2  
    # Generate a line along v
    t = np.linspace(-8, 8, 100)  # Range for plotting
    line_x = center[0] + t * v[0]
    line_y = center[1] + t * v[1]
    plt.plot(line_x, line_y, linestyle="--", color=color, label=label)

# Plot data points
plt.scatter(class_1[:, 0], class_1[:, 1], alpha=0.5, label="Class 1", color='lightblue')
plt.scatter(class_2[:, 0], class_2[:, 1], alpha=0.5, label="Class 2", color='lightcoral')

# Plot the original projection direction
plot_projection_line(v, "gray", "Original Projection v")

# Plot the rotated projection direction
plot_projection_line(v_new, "purple", f"Rotated Projection v' ({theta_deg}°)")

# Plot the optimal LDA projection direction
plot_projection_line(v_best, "green", "Best LDA Direction")


def plot_projection_line(mu_1,mu_2,v,arrow_color):
    # Project means onto v
    proj_mu_1 = np.dot(mu_1, v)
    proj_mu_2 = np.dot(mu_2, v)
    # Plot projected means
    proj_1 = proj_mu_1 * v
    proj_2 = proj_mu_2 * v
    formatted_v = ", ".join(f"{x:.2f}" for x in v)
    plt.scatter(*proj_1, color='blue', marker='o', s=100, edgecolors="black", label= f"Proj Mean 1 on [{formatted_v}]" )
    plt.scatter(*proj_2, color='coral', marker='o', s=100, edgecolors="black", label= f"Proj Mean 2 on [{formatted_v}]")

    # Draw arrows showing class separation
    plt.arrow(proj_1[0], proj_1[1], (proj_2 - proj_1)[0], (proj_2 - proj_1)[1], 
            head_width=0.3, head_length=0.3, fc=arrow_color, ec=arrow_color)

plot_projection_line(mu_1,mu_2,v,'black')
plot_projection_line(mu_1,mu_2,v_new,'purple')
plot_projection_line(mu_1,mu_2,v_best,'green')

# Labels and legend
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.title("Rotated Projection Line in LDA")
plt.grid()
plt.show()
