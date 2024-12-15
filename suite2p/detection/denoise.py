import numpy as np
import pywt
from typing import List
import time
from sklearn.decomposition import PCA
from ..registration.nonrigid import make_blocks, spatial_taper
def pca_denoise(mov: np.ndarray, block_size: List, n_comps_frac: float, wavelet: str = 'db1'):
    t0 = time.time()
    nframes, Ly, Lx = mov.shape
    yblock, xblock, _, block_size, _ = make_blocks(Ly, Lx, block_size=block_size)

    mov_mean = mov.mean(axis=0)
    mov -= mov_mean

    coeffs_all = []
    for frame in mov:
        coeffs = pywt.wavedec2(frame, wavelet, mode='periodization')
        coeffs_all.append(coeffs)
    mov_wavelet = np.stack([pywt.waverec2([coeffs[0]] + [None] * (len(coeffs) - 1), wavelet) for coeffs in coeffs_all])

    nblocks = len(yblock)
    Lyb, Lxb = block_size
    n_comps = int(min(min(Lyb * Lxb, nframes), min(Lyb, Lxb) * n_comps_frac))
    maskMul = spatial_taper(Lyb // 4, Lyb, Lxb)
    norm = np.zeros((Ly, Lx), np.float32)
    reconstruction = np.zeros_like(mov_wavelet)
    block_re = np.zeros((nblocks, nframes, Lyb * Lxb))

    for i in range(nblocks):
        block = mov_wavelet[:, yblock[i][0]:yblock[i][-1],
                            xblock[i][0]:xblock[i][-1]].reshape(-1, Lyb * Lxb)
        model = PCA(n_components=n_comps, random_state=0).fit(block)
        block_re[i] = (block @ model.components_.T) @ model.components_
        norm[yblock[i][0]:yblock[i][-1], xblock[i][0]:xblock[i][-1]] += maskMul

    block_re = block_re.reshape(nblocks, nframes, Lyb, Lxb)
    block_re *= maskMul
    for i in range(nblocks):
        reconstruction[:, yblock[i][0]:yblock[i][-1],
                       xblock[i][0]:xblock[i][-1]] += block_re[i]
    reconstruction /= norm

    reconstruction += mov_mean
    final_reconstruction = np.stack(
        [pywt.waverec2([pywt.wavedec2(reconstruction[frame], wavelet, mode='periodization')[0]] + [None] * (len(coeffs_all[0]) - 1), wavelet)
         for frame in range(nframes)]
    )

    print("Denoised with wavelet and PCA in %0.2f sec." % (time.time() - t0))
    return final_reconstruction
