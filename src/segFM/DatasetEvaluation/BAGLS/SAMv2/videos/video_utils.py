import numpy as np
import cv2


def dice_score(img, seg) -> float:
    """
    Computes the dice score between two binary images.
    The dice score is a measure of overlap between two sets.
    It is represented by a value between 0 and 1, where 1 indicates perfect overlap.
    This can be used to compare a predicted segmentation with a ground truth segmentation.
    """

    print(np.unique(img))
    print(np.unique(seg))

    overlap = np.count_nonzero(img * seg)
    union = np.count_nonzero(img) + np.count_nonzero(seg)

    #No overlap is still considered a perfect score if all pixels are 0
    if union == 0:
        return 1.0

    return (2.0 * overlap + 1e-6) / (union + 1e-6) # to avoid possible division by zero


def random_point_prompts_from_gt(gt, positive=1, negative=0, negative_size=20):
    """
    Generates a random point prompt from the ground truth segmentation mask.
    The point is chosen randomly from the non-zero pixels of the mask.

    Parameters:
        gt (numpy.ndarray): The ground truth segmentation mask.
        positive (int): The number of positive points to generate.
        negative (int): The number of negative points to generate.
        negative_size (int): The maximum distance a negative point can be from the mask.
    Returns:
        tuple: A tuple containing two numpy arrays:
            - prompts: The coordinates of the points.
            - labels: The labels of the points (1 for positive, 0 for negative).
    """
    # Binarize the ground truth mask
    gt[gt < 130] = 0
    gt[gt >= 130] = 255

    prompts = []
    labels = []

    ##### Positive prompts #####
    if positive > 0:
        # Perform binary erosion to slightly reduce the size of the mask
        eroded = cv2.erode(gt.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)

        # Get the indices of the non-zero pixels
        indices_pos = np.argwhere(eroded > 0)

        for i in range(positive):
            if len(indices_pos) == 0:
                continue
            # Choose a random index
            random_index = np.random.choice(len(indices_pos))

            # Get the coordinates of the random point
            random_point = indices_pos[random_index]

            # Reverse the order to (x, y) and append the point and label to the lists
            prompts.append(random_point[::-1])
            labels.append(1)

    ##### Negative prompts #####
    if negative > 0:
        # Perform binary dilation to slightly increase the size of the mask
        dilated = cv2.dilate(gt.astype(np.uint8), np.ones((negative_size, negative_size), np.uint8), iterations=1)

        # Subtract the original mask to only keep the pixels outside the mask
        negative_mask = gt - dilated

        # Get the indices of the non-zero pixels
        indices = np.argwhere(negative_mask > 0)

        for i in range(negative):
            if len(indices) == 0:
                return None, None
            
            # Choose a random index
            random_index = np.random.choice(len(indices))

            # Get the coordinates of the random point
            random_point = indices[random_index]

            # Reverse the order to (x, y) and append the point and label to the lists
            prompts.append(random_point[::-1])
            labels.append(0)

    return np.array(prompts), np.array(labels)