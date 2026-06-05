import matplotlib.pyplot as plt
# Data
batch_sizes = [128, 256, 512, 1024]
batch_perf = [49.5, 52.1, 54.3, 55.9]
step_sizes = [1000, 2000, 4000, 8000]
step_perf = [49.8, 52.0, 53.8, 55.3]
num_crops = [2, 4, 8, 16]
crop_perf = [47.1, 52.0, 54.2, 54.8]
# Plot
fig, axs = plt.subplots(1, 3, figsize=(10, 3))
# Batch size subplot
axs[0].plot(batch_sizes, batch_perf, marker='o', color='steelblue')
axs[0].set_title('Batch Size Influence on Performance', fontsize=9, fontweight='bold')
axs[0].set_xlabel('Batch Size')
axs[0].set_ylabel('Performance (%)')
# Step size subplot
axs[1].plot(step_sizes, step_perf, marker='s', linestyle='--', color='green')
axs[1].set_title('Step Size Influence on Performance', fontsize=9, fontweight='bold')
axs[1].set_xlabel('Step Size')
axs[1].set_ylabel('Performance (%)')
# Number of crops subplot
axs[2].plot(num_crops, crop_perf, marker='^', linestyle='-.', color='firebrick')
axs[2].set_title('Number of Crops Influence on Performance', fontsize=9, fontweight='bold')
axs[2].set_xlabel('Number of Crops')
axs[2].set_ylabel('Performance (%)')
# Tidy up
for ax in axs:
    ax.grid(True)
plt.tight_layout()
plt.show()
plt.savefig("performance_plots_high_res.pdf", format='pdf', dpi=300)