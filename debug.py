# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
metrics = np.load(
    "results/ETTm1_96_96_PaiFilter_ETTm1_sl96_pl96_embed128_hidden256_bs32_lr0.01_drop0_0/metrics.npy"
)
preds = np.load(
    "results/ETTm1_96_96_PaiFilter_ETTm1_sl96_pl96_embed128_hidden256_bs32_lr0.01_drop0_0/pred.npy"
)
truth = np.load(
    "results/ETTm1_96_96_PaiFilter_ETTm1_sl96_pl96_embed128_hidden256_bs32_lr0.01_drop0_0/true.npy"
)


# %%
plt.figure(figsize=(10, 6))
plt.imshow(preds[0], aspect="auto", cmap="viridis")
plt.colorbar(label="Value")
plt.title("Predictions Visualization")
plt.xlabel("Time Steps")
plt.ylabel("Features")
plt.show()

plt.figure(figsize=(10, 6))
plt.imshow(truth[0], aspect="auto", cmap="viridis")
plt.colorbar(label="Value")
plt.title("Predictions Visualization")
plt.xlabel("Time Steps")
plt.ylabel("Features")
plt.show()

# %%

# %%
