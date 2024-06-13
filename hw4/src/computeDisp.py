import numpy as np
import cv2.ximgproc as xip  # type: ignore (ignore warning)


def census_cost(local_binary_L, local_binary_R):
    disparity = np.sum(np.abs(local_binary_L - local_binary_R), axis=1)
    return disparity


def computeDisp(Il, Ir, max_disp):
    h, w, ch = Il.shape
    labels = np.zeros((h, w), dtype=np.float32)
    Il = Il.astype(np.float32)
    Ir = Ir.astype(np.float32)

    # pad the image
    window_size = 3
    pad_size = window_size // 2

    img_L = np.pad(Il, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode="constant", constant_values=0)
    img_R = np.pad(Ir, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode="constant", constant_values=0)

    # >>> Cost Computation
    # TODO: Compute matching cost
    # [Tips] Census cost = Local binary pattern -> Hamming distance
    # [Tips] Set costs of out-of-bound pixels = cost of closest valid pixel
    # [Tips] Compute cost both Il to Ir and Ir to Il for later left-right consistency

    # >>> Cost Aggregation
    # TODO: Refine the cost according to nearby costs
    # [Tips] Joint bilateral filter (for the cost of each disparty)

    # store the matching costs for left-to-right and right-to-left disparities.
    cost_list_L = np.zeros((h, w, max_disp), dtype=np.float32)
    cost_list_R = np.zeros((h, w, max_disp), dtype=np.float32)

    local_binary_IL, local_binary_IR = [], []
    for i in range(h):
        for j in range(w):
            patch_L = img_L[i : i + window_size, j : j + window_size, :].copy()
            patch_R = img_R[i : i + window_size, j : j + window_size, :].copy()
            for k in range(ch):
                center_L = patch_L[pad_size, pad_size, k]
                center_R = patch_R[pad_size, pad_size, k]
                patch_L[:, :, k] = (patch_L[:, :, k] >= center_L).astype(np.float32)
                patch_R[:, :, k] = (patch_R[:, :, k] >= center_R).astype(np.float32)
            local_binary_IL.append(patch_L.flatten())
            local_binary_IR.append(patch_R.flatten())

    local_binary_IL = np.array(local_binary_IL).reshape(h, w, -1)  # (h, w, window_size*window_size*ch)
    local_binary_IR = np.array(local_binary_IR).reshape(h, w, -1)  # (h, w, window_size*window_size*ch)

    for i in range(h):
        for j in range(w):
            if j < max_disp - 1:
                local_binary_L = local_binary_IL[i, j][np.newaxis, :]
                local_binary_R = np.flip(local_binary_IR[i, : j + 1], axis=0)
                disparity = census_cost(local_binary_L, local_binary_R)
                cost_list_L[i, j, : j + 1] = disparity
                cost_list_L[i, j, j + 1 :] = disparity[-1]
            else:
                local_binary_L = local_binary_IL[i, j][np.newaxis, :]
                local_binary_R = np.flip(local_binary_IR[i, (j - max_disp + 1) : j + 1], axis=0)
                disparity = census_cost(local_binary_L, local_binary_R)
                cost_list_L[i, j, :] = disparity

            if j + max_disp > w:
                local_binary_L = local_binary_IL[i, j:w]
                local_binary_R = local_binary_IR[i, j][np.newaxis, :]
                disparity = census_cost(local_binary_L, local_binary_R)
                cost_list_R[i, j, : w - j] = disparity
                cost_list_R[i, j, w - j :] = disparity[-1]
            else:
                local_binary_L = local_binary_IL[i, j : j + max_disp]
                local_binary_R = local_binary_IR[i, j][np.newaxis, :]
                disparity = census_cost(local_binary_L, local_binary_R)
                cost_list_R[i, j, :] = disparity

    # Apply joint bilateral filter
    for d in range(max_disp):
        cost_list_L[:, :, d] = xip.jointBilateralFilter(Il, cost_list_L[:, :, d], 20, 10, 10)
        cost_list_R[:, :, d] = xip.jointBilateralFilter(Ir, cost_list_R[:, :, d], 20, 10, 10)

    # >>> Disparity Optimization
    # TODO: Determine disparity based on estimated cost.
    # [Tips] Winner-take-all
    winner_L = np.argmin(cost_list_L, axis=2)
    winner_R = np.argmin(cost_list_R, axis=2)

    # >>> Disparity Refinement
    # TODO: Do whatever to enhance the disparity map
    # [Tips] Left-right consistency check -> Hole filling -> Weighted median filtering

    # Inconsistent disparities are marked as -1 and then filled by the closest valid disparities
    for i in range(h):
        for j in range(w):
            if j - winner_L[i, j] >= 0 and winner_L[i, j] == winner_R[i, j - winner_L[i, j]]:
                continue
            else:
                winner_L[i, j] = -1

    for i in range(h):
        for j in range(w):
            if winner_L[i, j] == -1:
                left_disparity = np.inf
                right_disparity = np.inf

                l_idx = j - 1
                while l_idx >= 0 and winner_L[i, l_idx] == -1:
                    l_idx -= 1
                if l_idx >= 0:
                    left_disparity = winner_L[i, l_idx]

                r_idx = j + 1
                while r_idx < w and winner_L[i, r_idx] == -1:
                    r_idx += 1
                if r_idx < w:
                    right_disparity = winner_L[i, r_idx]

                winner_L[i, j] = min(left_disparity, right_disparity)

    labels = xip.weightedMedianFilter(Il.astype(np.uint8), winner_L.astype(np.uint8), 18, 1)
    return labels.astype(np.uint8)
