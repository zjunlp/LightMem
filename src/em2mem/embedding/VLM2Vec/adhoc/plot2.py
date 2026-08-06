import matplotlib.pyplot as plt
import numpy as np

# Data
modalities = ["Image", "VisDoc", "Video"]
lora_8 = [62.7, 52.5, 32.4]
lora_16 = [63.2, 52.6, 33.5]
lora_32 = [60.0, 52.1, 32.7]

# Bar placement
x = np.array([0, 1, 2])  # modality positions
bar_width = 0.2
offset = 0.24  # control spacing between LoRA bars

# Font settings
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 14

# Create plot
plt.figure(figsize=(7, 6))
bars1 = plt.bar(x - offset, lora_8, bar_width, label='LoRA 8', color='#1f77b4')
bars2 = plt.bar(x, lora_16, bar_width, label='LoRA 16', color='#ff7f0e')
bars3 = plt.bar(x + offset, lora_32, bar_width, label='LoRA 32', color='#2ca02c')

# Axes and labels
plt.xticks(x, modalities, fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel("Modality", fontsize=18)
plt.ylabel("Performance", fontsize=18)
plt.title("Performance under Different LoRA Ranks", fontsize=18)
plt.ylim(30, 70)

# Annotate bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.5,
                 f'{height:.1f}', ha='center', va='bottom', fontsize=14)

# Legend without frame
plt.legend(frameon=False, fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()

# Save as PDF
plt.savefig("lora_rank_comparison_y30_wider.pdf", format='pdf', dpi=300)
plt.show()
