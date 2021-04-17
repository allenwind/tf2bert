import numpy as np
import matplotlib.pyplot as plt

# 验证两种掩码是否正确

def lm_mask_plot(seq_len):
    indices = np.arange(seq_len)
    mask = indices[None, :] <= indices[:, None]
    mask = np.float32(mask)
    mask = - (1 - mask[None, None]) * 1e12
    plt.imshow(mask[0][0])
    plt.show()

def uni_lm_mask_plot(segments):
    indices = np.cumsum(segments, axis=1)
    mask = indices[:, None, :] <= indices[:, :, None]
    mask = np.float32(mask)
    mask = -(1 - mask[:, None]) * 1e12
    plt.imshow(mask[0][0])
    plt.show()

if __name__ == "__main__":
    lm_mask_plot(100)
    segments = np.array([[0] * 60 + [1] * 40])
    uni_lm_mask_plot(segments)
