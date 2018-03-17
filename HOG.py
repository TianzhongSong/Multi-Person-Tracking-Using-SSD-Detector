import cv2
import numpy as np
import random

random.seed(0)


def atoi(text):
    return int(text) if text.isdigit() else text


def histogram2(angles, magnitudes):
    # Compute the histogram
    bins = [0, 20, 40, 60, 80, 100, 120, 140, 160]
    h = np.zeros(len(bins))

    for i in range(np.shape(angles)[0]):
        for j in range(np.shape(angles)[1]):
            my_list = [e for e, a in zip(range(len(bins)), bins) if a < angles[i][j]]

            index = 0
            if my_list:
                index = (my_list[-1] + 1) % len(bins)

            difference = np.abs(bins[index] - (angles[i][j] % 160))
            if difference == 160:
                proportion = 1
            else:
                proportion = difference / 20

            values = magnitudes[i][j] * proportion, magnitudes[i][j] * (1 - proportion)

            if angles[i][j] < 160:
                h[index - 1] += values[0]
                h[index] += values[1]
            else:
                h[index - 1] += values[1]
                h[index] += values[0]

    return h


def histogram(angles, magnitudes):
    # [0, 20, 40, 60, 80, 100, 120, 140, 160]
    h = np.zeros(10, dtype=np.float32)

    for i in range(angles.shape[0]):
        for j in range(angles.shape[1]):
            angles[i, j] = 160
            index_1 = int(angles[i, j] // 20)
            index_2 = int(angles[i, j] // 20 + 1)

            proportion = (index_2 * 20 - angles[i, j]) / 20

            value_1 = proportion * magnitudes[i, j]
            value_2 = (1 - proportion) * magnitudes[i, j]

            h[index_1] += value_1
            h[index_2] += value_2

    h[0] += h[-1]
    return h[0:9]


def make_cells(angles, magnitudes, cell_size):
    cells = []
    for i in range(0, np.shape(angles)[0], cell_size):
        row = []
        for j in range(0, np.shape(angles)[1], cell_size):
            row.append(np.array(
                histogram(angles[i:i + cell_size, j:j + cell_size], magnitudes[i:i + cell_size, j:j + cell_size]),
                dtype=np.float32))
        cells.append(row)

    return np.array(cells, dtype=np.float32)


def make_blocks(block_size, cells):
    before = int(block_size / 2)
    after = int(block_size / 2)

    if block_size % 2 != 0:
        after = after + 1

    first_stop = before
    second_stop = before

    if np.shape(cells)[0] % block_size == 0:
        first_stop = first_stop - 1

    if np.shape(cells)[1] % block_size == 0:
        second_stop = second_stop - 1

    blocks = []
    for i in range(int(block_size / 2.0), np.shape(cells)[0] - first_stop):
        for j in range(int(block_size / 2.0), np.shape(cells)[1] - second_stop):
            blocks.append(np.array(cells[i - before:i + after, j - before:j + after].flatten()))

    return blocks


def normalize_L1(block, threshold):
    norm = np.sum(block) + threshold
    if norm != 0:
        return block / norm
    else:
        return block


def normalize_L1_sqrt(block, threshold):
    norm = np.sum(block) + threshold
    if norm != 0:
        return np.sqrt(block / norm)
    else:
        return block


def normalize_L2(block, threshold):
    norm = np.sqrt(np.sum(block * block) + threshold * threshold)
    if norm != 0:
        return block / norm
    else:
        return block


def normalize_L2_Hys(block, threshold):
    norm = np.sqrt(np.sum(block * block) + threshold * threshold)

    if norm != 0:
        block_aux = block / norm
        block_aux[block_aux > 0.2] = 0.2
        norm = np.sqrt(np.sum(block_aux * block_aux) + threshold * threshold)
        if norm != 0:
            return block_aux / norm
        else:
            return block_aux
    else:
        return block


def normalize(block, type_norm, threshold=0):
    if type_norm == 0:
        return normalize_L2(block, threshold)
    elif type_norm == 1:
        return normalize_L2_Hys(block, threshold)
    elif type_norm == 2:
        return normalize_L1(block, threshold)
    elif type_norm == 3:
        return normalize_L1_sqrt(block, threshold)


def hog(img, cell_size=6, block_size=3, type_norm=0, all_norms=False):
    # Gamma correction : gamma = 0.2
    img = np.power(img, 0.2, dtype=np.float32)

    # Calculate gradient
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)

    # Gradient magnitude and direction (in degrees)
    magnitudes, angles = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    max_norm = np.argmax(magnitudes, axis=2)

    # For each pixel, we store the angle with the biggest magnitude
    m, n = np.shape(max_norm)
    I, J = np.ogrid[:m, :n]
    max_angles = angles[I, J, max_norm]
    max_magnitudes = magnitudes[I, J, max_norm]

    # Convert angles to be in range 0-180
    max_angles = max_angles % 180

    # Obtain the histogram for each region
    cells = make_cells(max_angles, max_magnitudes, cell_size)

    # Append the cells into blocks: 180 descriptors with 81 elements
    blocks = make_blocks(block_size, cells)

    # We need to apply a Gaussian kernel
    # kernel = [-1, 0, 1]
    # sigma = 0.5 * block_size
    # kernel = np.array([math.exp(-0.5 * (x ** 2 / sigma ** 2)) for x in kernel])
    # multiplier = np.append(np.append(np.ones(36) * kernel[0], np.ones(9) * kernel[1]), np.ones(36) * kernel[2])
    # blocks = [b * multiplier for b in blocks]

    # Now we need to normalize
    if all_norms:
        blocks_norm0 = np.concatenate([normalize(b, 0) for b in blocks])
        blocks_norm1 = np.concatenate([normalize(b, 1) for b in blocks])
        blocks_norm2 = np.concatenate([normalize(b, 2) for b in blocks])
        blocks_norm3 = np.concatenate([normalize(b, 3) for b in blocks])

        return blocks_norm0, blocks_norm1, blocks_norm2, blocks_norm3
    else:
        blocks = np.concatenate([normalize(b, type_norm) for b in blocks])
        # Concatenate all the blocks and this is the final descriptor
        return blocks
