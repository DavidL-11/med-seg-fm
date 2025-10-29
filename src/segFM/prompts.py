import matplotlib.pyplot as plt
import numpy as np
import cv2

from segFM import utils
from segFM.BOB.src.BOB.inference.utils import Prompt

"""
This file contains the Prompt class used for segmentation tasks, 
as well as functions to generate prompts from ground truth segmentation masks.
"""


def center_of_mass_point_prompt(bin_img, bbsize, show_plot=False):
    """
    Creates a bounding box and a point prompt from the ground truth segmentation mask.
    The bounding box is created around the white regions of the binary image.
    The point prompt is the center of mass of the white regions in the bounding box.

    Parameters:
        bin_img (numpy.ndarray): The binary image to create the bounding box from.
        bbsize (int): The padding size to be added to the bounding box.
    Returns:
        tuple: The coordinates of the bounding box (x, y, w, h).
        numpy.ndarray: The coordinates of a point inside the bounding box.
        numpy.ndarray: The label of the point (1 for foreground, 0 for background).
    """
    # Find the bounding box of the white regions in the segmented image
    x, y, w, h = cv2.boundingRect(bin_img)

    # Increase the bounding box by bbsize pixels
    x = max(0, x - bbsize)
    y = max(0, y - bbsize)
    w = min(bin_img.shape[1], w + 2 * bbsize)
    h = min(bin_img.shape[0], h + 2 * bbsize)

    # Get the moments within the bounding box
    M = cv2.moments(bin_img[y : y + h, x : x + w])

    # Initialize the center of mass coordinates
    cX, cY = 0, 0

    # Calculate the center of mass using the moments
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX = int(w / 2)
        cY = int(h / 2)

    # Find a point inside the area of the image to be used as a point prompt
    input_point = np.array([[x + cX, y + cY]])
    input_label = np.array([1])

    # Plot the generated box/point prompts
    if show_plot:
        cv2.rectangle(bin_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.circle(bin_img, (x + cX, y + cY), 5, (0, 255, 0), -1)
        plt.imshow(bin_img, cmap="gray")
        plt.title("Bounding Box")
        plt.axis("off")
        plt.show()

    # If the bounding box is empty (emptry ground truth), no prompt should be used
    if bin_img[y : y + h, x : x + w].sum() == 0:
        raise ValueError("The bounding box is empty. No prompt should be used.")

    return Prompt(
        box=None,
        point=input_point,
        label=input_label,
        quality_score=1.0,
        generated_by="center_of_mass",
    )


def canny_box_prompt(img):
    img_blur = cv2.GaussianBlur(img, (3, 3), 0)
    img_blur[img_blur < 10] = 0
    # Canny Edge Detection
    edges = cv2.Canny(
        image=img_blur, threshold1=50, threshold2=100
    )  # Canny Edge Detection

    # Remove contours longer than 100 pixels
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank image to draw the contours
    blank = np.zeros_like(edges)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 20 and len(contour) > 40:
            cv2.drawContours(blank, [contour], -1, (255, 255, 255))

    # Find the bounding box of the newly created image
    x, y, w, h = cv2.boundingRect(blank)

    return Prompt(box=np.array([x, y, x + w, y + h]), generated_by="canny_box_prompt")


def harris_corners_point_prompt(img):
    # Convert the image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_blur = cv2.GaussianBlur(img, (3, 3), 0)
    img_blur[img_blur < 10] = 0

    # Harris Corner Detection
    corners = cv2.cornerHarris(img_blur, blockSize=5, ksize=7, k=0.15)

    # Dilate the corners to enhance them
    corners = cv2.dilate(corners, None)

    # Threshold to get the best corners
    threshold = 0.01 * corners.max()
    corners[corners > threshold] = 255
    corners[corners <= threshold] = 0
    corners = corners.astype(np.uint8)

    # Find the bounding box of the corners
    x, y, w, h = cv2.boundingRect(corners)

    # Find the minimum value within the bounding box and save its coordinates
    min_coords = np.unravel_index(np.argmin(img_blur[y : y + h, x : x + w]), (h, w))
    # Adjust coordinates to the original image
    min_coords = (min_coords[1] + x, min_coords[0] + y)

    return Prompt(
        point=np.array([min_coords]),
        label=np.array([1]),
        generated_by="harris_corners_point_prompt",
    )


def canny_point_prompt(img):
    # Convert the image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img, (3, 3), 0)
    img_blur[img_blur < 10] = 0
    # Canny Edge Detection
    edges = cv2.Canny(
        image=img_blur, threshold1=50, threshold2=100
    )  # Canny Edge Detection

    # Remove contours longer than 100 pixels
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blank = np.zeros_like(edges)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 20 and len(contour) > 40:
            cv2.drawContours(blank, [contour], -1, (255, 255, 255))

    # Find the bounding box of the newly created image
    x, y, w, h = cv2.boundingRect(blank)

    # Find the minimum value within the bounding box and save its coordinates
    min_coords = np.unravel_index(np.argmin(img_blur[y : y + h, x : x + w]), (h, w))
    # Adjust coordinates to the original image
    min_coords = (min_coords[1] + x, min_coords[0] + y)

    return Prompt(
        point=np.array([min_coords]),
        label=np.array([1]),
        generated_by="canny_point_prompt",
    )


def random_point_prompt_from_gt(gt):
    """
    Generates a random point prompt from the ground truth segmentation mask.
    The point is chosen randomly from the non-zero pixels of the mask.

    Parameters:
        gt (numpy.ndarray): The ground truth segmentation mask.
    Returns:
        numpy.ndarray: The coordinates of the random point.
        numpy.ndarray: The label of the point (1 for foreground, 0 for background).
    """
    # Perform binary erosion to slightly reduce the size of the mask
    eroded = cv2.erode(gt.astype(np.uint8), np.ones((12, 12), np.uint8), iterations=1)
    if eroded.sum() == 0:  # If the erosion removed all pixels, try a smaller kernel
        eroded = cv2.erode(gt.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1)
    if (
        eroded.sum() == 0
    ):  # If the erosion still removed all pixels, use the original mask
        eroded = gt

    # Get the indices of the non-zero pixels
    indices = np.argwhere(gt > 0)

    # Choose a random index
    random_index = np.random.choice(len(indices))

    # Get the coordinates of the random point
    random_point = indices[random_index]

    # Reverse the order to (x, y)
    random_point = random_point[::-1]

    return Prompt(
        point=np.array([random_point]),
        label=np.array([1]),
        quality_score=1.0,
        generated_by="random_point_from_gt",
    )


def random_prompts_from_binary_gt(gt, dataset, npos=1, nneg=0, neg_size=10):
    """
    Generates prompts from ground truth data.
    """
    # Binarize the ground truth mask
    gt[gt < 130] = 0
    gt[gt >= 130] = 255

    prompts = []
    labels = []

    ##### Positive prompts #####
    if npos > 0:
        # Perform binary erosion to slightly reduce the size of the mask
        eroded = cv2.erode(gt.astype(np.uint8), np.ones((7, 7), np.uint8), iterations=1)
        if eroded.sum() == 0:  # If the erosion removed all pixels, try a smaller kernel
            eroded = cv2.erode(
                gt.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1
            )
        if (
            eroded.sum() == 0
        ):  # If the erosion still removed all pixels, use the original mask
            eroded = gt

        # Get the indices of the non-zero pixels
        indices_pos = np.argwhere(eroded > 0)

        for i in range(npos):
            if len(indices_pos) == 0:
                return Prompt()
            # Choose a random index
            random_index = np.random.choice(len(indices_pos))

            # Get the coordinates of the random point
            random_point = indices_pos[random_index]

            # Reverse the order to (x, y) and append the point and label to the lists
            prompts.append(random_point[::-1])
            labels.append(1)

    ##### Negative prompts #####
    if nneg > 0:
        # Perform binary dilation to slightly increase the size of the mask
        dilated = cv2.dilate(
            gt.astype(np.uint8), np.ones((neg_size, neg_size), np.uint8), iterations=1
        )

        # Subtract the original mask to only keep the pixels outside the mask
        negative_mask = gt - dilated

        # Get the indices of the non-zero pixels
        indices = np.argwhere(negative_mask > 0)

        for i in range(nneg):
            if len(indices) == 0:
                return Prompt()

            # Choose a random index
            random_index = np.random.choice(len(indices))

            # Get the coordinates of the random point
            random_point = indices[random_index]

            # Reverse the order to (x, y) and append the point and label to the lists
            prompts.append(random_point[::-1])
            labels.append(0)

    return Prompt(
        point=np.array(prompts),
        label=np.array(labels),
        color=255,
        class_label=dataset.color_to_label.get(255, "Unknown"),
        quality_score=1.0,
        generated_by="random_prompts_from_gt",
    )


def point_prompt_darkest_pixel(img, gt, bbsize):
    """
    Generates a point prompt from the darkest pixel in the bounding box of the ground truth segmentation mask.
    """

    p = box_prompt_from_gt(gt, bbsize)
    if p.is_empty():
        return Prompt()
    
    x1, y1, x2, y2 = p.box

    # Get the darkest pixel in the bounding box

    box_img = gt[y1:y2, x1:x2]
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(box_img)
    coords = (min_loc[0] + x1, min_loc[1] + y1)

    return Prompt(
        point=np.array([coords]),
        label=np.array([1]),
        generated_by="point_prompt_darkest_pixel",
    )


def generate_point_prompt_at_rim(gt):
    """
    Generates a point prompt at the rim of the ground truth segmentation mask.
    The point is chosen randomly from the outermost pixels of the mask.
    """

    # Perform binary erosion to slightly reduce the size of the mask
    gt_small = cv2.erode(gt.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1)
    gt_rim = cv2.subtract(gt.astype(np.uint8), gt_small)

    # Get the indices of the non-zero pixels
    indices = np.argwhere(gt_rim > 0)

    if len(indices) == 0:  # If the rim is empty, return an empty prompt
        return Prompt()

    # Choose a random index
    random_index = np.random.choice(len(indices))

    # Get the coordinates of the random point
    random_point = indices[random_index]

    # Reverse the order to (x, y)
    random_point = random_point[::-1]

    return Prompt(
        point=np.array([random_point]),
        label=np.array([1]),
        generated_by="generate_point_prompt_at_rim",
    )


def box_prompt_from_gt(gt, dataset, bbsize: int = 0) -> Prompt:
    """
    Creates a bounding box from the ground truth segmentation mask.
    The bounding box is created around the white regions of the binary image.

    Parameters:
        gt (numpy.ndarray): The binary image to create the bounding box from.
        bbsize (int): The padding size to be added to the bounding box. Can be an integer or a percentage string (e.g., "10%").
    Returns:
        numpy.ndarray: The coordinates of the bounding box (x1, y1, x2, y2).
    """
    # Find the bounding box of the white regions in the segmented image
    x, y, w, h = cv2.boundingRect(gt)

    if w == 0 or h == 0:  # If the bounding box is empty, return None
        return Prompt()
    
    if type(bbsize) == str and bbsize.endswith("%"):
        # Convert percentage to integer
        percent = float(bbsize.strip("%")) / 100.0
        # Increase the bounding box by the given percentage of the width and height
        x = max(0, int(x - w * percent))
        y = max(0, int(y - h * percent))
        w = min(gt.shape[1], int(w + 2 * w * percent))
        h = min(gt.shape[0], int(h + 2 * h * percent))
    else:
        # Increase the bounding box by bbsize pixels
        x = max(0, x - bbsize)
        y = max(0, y - bbsize)
        w = min(gt.shape[1], w + 2 * bbsize)
        h = min(gt.shape[0], h + 2 * bbsize)

    return Prompt(box=np.array([x, y, x + w, y + h]), 
                  color=255,
                  generated_by="box_prompt_from_gt")


def single_box_prompt_3d(gt, plot_prompt=False):
    """
    Returns a bounding box for the given ground truth.
    Format is z, [x1, y1, x2, y2]
    The gt is expected to be a 3D numpy array.
    The bounding box is a 2D array obtained by slicing the mask along the z-axis at a good coordinate
    and then finding the bounding box of the ground truth.
    """
    # Find a good z coordinate
    z = np.argmax(np.sum(gt, axis=(1, 2)))
    # Find the bounding box in the xy plane
    x, y = np.where(gt[z] > 0)

    x1, y1, w, h = cv2.boundingRect(gt[z])

    if plot_prompt:  # Plot the bounding box
        rect = cv2.rectangle(gt[z].copy(), (x1, y1), (x1 + w, y1 + h), (255, 0, 0), 2)
        plt.imshow(rect, cmap="gray")
        plt.show()

    return [Prompt(
        box=np.array([x1, y1, x1 + w, y1 + h]), obj_id=1, z=z, generated_by="single_box_prompt_3d"
    )]


def multicolor_box_prompt_3d(gt, dataset, plot_prompt=False, img=None):
    """
    Returns as many bounding boxes as there are objects in the ground truth.
    Format is z, List([x1, y1, x2, y2])
    The gt is expected to be a 3D numpy array.
    The bounding box is a 2D array obtained by slicing the mask along the z-axis at a good coordinate
    and then finding the bounding box of the ground truth.
    """
    assert len(gt.shape) == 3, "The ground truth mask must be a 3D numpy array."

    object_colors = np.unique(gt)  # Retrieve all the unique colors in the mask
    # Remove the background color (0)
    object_colors = object_colors[object_colors > 0]

    boxes = []

    # Iterate over the unique colors and find the bounding box for each color
    for color in object_colors:
        color_mask = (gt == color).astype(np.uint8)

        # Find the slice with the most pixels
        z = np.argmax(np.sum(color_mask, axis=(1, 2)))
        # Find the bounding box of the color mask
        x1, y1, w, h = cv2.boundingRect(color_mask[z])
        # Add it to the dictionary
        boxes.append(
            (
                np.array([x1, y1, x1 + w, y1 + h]),
                z,
                color
            )
        )

    if plot_prompt:  # Plot the bounding box
        for box, z, color in boxes:
            x1, y1, x2, y2 = box
            if img is not None:
                z_slice = img[z].copy()
            else:
                z_slice = gt[z].copy()
            # Normalize it to 0-255 for visualization
            z_slice = cv2.normalize(z_slice, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            rect = cv2.rectangle(z_slice, (x1, y1), (x2, y2), (255, 0, 0), 2)
            plt.imshow(rect, cmap="gray")
            plt.show()

    ps = list(
        map(
            lambda b: Prompt(
                box=b[0],
                z=b[1],
                class_label=dataset.color_to_label[b[2]],
                class_id=dataset.color_to_id[b[2]],
                color=b[2],
                generated_by="multicolor_box_prompt_3d",
            ),
            boxes,
        )
    )
    # Assign a unique object ID to each prompt
    for i, p in enumerate(ps):
        p.obj_id = i

    return ps


def get_prompt(
    gt, image, dataset, bbsize=0, mode="point", prompt_finder="random", n_pos=1, n_neg=0, yolo=None
) -> Prompt:
    """
    Generates a prompt for a segmentation model based on the ground truth segmentation mask and the input image.
    The prompt can be a bounding box, a point, or both, depending on the mode specified.
    The prompt finder determines how the prompt is generated, such as using the center of mass or random points from the ground truth.
    """
    box_prompt = box_prompt_from_gt(gt, dataset, bbsize)
    if mode == "box":
        return box_prompt

    if prompt_finder == "random":  # Only prompt where n_pos and n_neg are used
        prompt = random_prompts_from_binary_gt(gt, dataset, npos=n_pos, nneg=n_neg, neg_size=50)
    elif prompt_finder == "center":
        prompt = center_of_mass_point_prompt(gt, bbsize)
    elif prompt_finder == "darkest":
        prompt = point_prompt_darkest_pixel(image, gt, bbsize)
    elif prompt_finder == "rim":
        prompt = generate_point_prompt_at_rim(gt)
    else:
        raise ValueError(
            "Invalid prompt finder. Use 'random', 'center', 'darkest', or 'rim'."
        )

    if mode == "point":
        return prompt
    elif mode == "both": # Combines the box prompt with the point prompt
        return Prompt(
            box=box_prompt.box,
            point=prompt.point,
            label=prompt.label,
            class_label=prompt.class_label,
            color=prompt.color,
            generated_by=prompt.generated_by,
        )
    else:
        raise ValueError("Invalid mode. Use 'box' or 'point' or 'both' mode.")


def get_multicolor_prompt(
    gt,
    image,
    dataset,
    bbsize=0,
    mode="point",
    prompt_finder="random",
    n_pos=1,
    n_neg=0,
    plot_prompts=False,
):
    """
    Generates a prompt for each object (= different color) in the 2D grayscale ground truth segmentation mask.
    The prompt can be a bounding box, a point, or both, depending on the mode specified.
    """
    if gt.ndim == 3:
        return get_labelmap_prompt(
            gt, image, dataset, bbsize=bbsize, mode=mode, prompt_finder=prompt_finder,
            n_pos=n_pos, n_neg=n_neg, plot_prompts=plot_prompts
        )
    elif gt.ndim != 2:
        raise ValueError(
            "Ground truth mask must be a 2D numpy array with shape (H, W)."
        )
    unique_colors = np.unique(gt)
    unique_colors = unique_colors[unique_colors > 0]  # Exclude background (0)

    # SAM2 can also accept multiple prompts for multiple objects by providing an array of boxes/points and labels
    prompts = []

    for color in unique_colors:
        # Create a binary mask for the current color
        color_gt = np.zeros_like(gt, dtype=np.uint8)
        color_gt[gt == color] = 255

        # Get the prompt for the current color
        prompt = get_prompt(
            gt=color_gt, 
            image=image, 
            dataset=dataset, 
            bbsize=bbsize, 
            mode=mode, 
            prompt_finder=prompt_finder, 
            n_pos=n_pos, 
            n_neg=n_neg
        )

        prompt.color = color  # Store the color of the object in the prompt
        prompt.class_label = dataset.color_to_label.get(color, "unknown")
        prompt.class_id = dataset.color_to_id.get(color, -1)
        prompts.append(prompt)

    # Plot prompts
    if plot_prompts:
        for prompt in prompts:
            img = image.copy()
            if prompt.is_empty():
                continue
            if prompt.box is not None:
                x1, y1, x2, y2 = prompt.box
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
            if prompt.point is not None:
                for point in prompt.point:
                    cv2.circle(img, tuple(point), 5, (255, 0, 255), -1)
            # Subplot image, object_gt and prompt
            plt.subplot(1, 2, 1)
            plt.imshow(img, cmap="gray")
            plt.title(f"Prompt for {prompt.class_label} (Color: {prompt.color})")
            plt.axis("off")
            plt.subplot(1, 2, 2)
            plt.imshow(gt==prompt.color, cmap="gray")
            plt.title("Ground Truth")
            plt.axis("off")
            plt.show()
    return prompts

def get_multiobject_multicolor_prompt(
    gt,
    image,
    dataset,
    bbsize=0,
    mode="point",
    prompt_finder="random",
    n_pos=1,
    n_neg=0,
    plot_prompts=False,
):
    """
    Generates a prompt for each object (= not connected component) in the ground truth segmentation mask.
    The prompt can be a bounding box, a point, or both, depending on the mode specified.
    """
    if gt.ndim != 2:
        raise ValueError(
            "Ground truth mask must be a 2D numpy array with shape (H, W)."
        )

    unique_colors = np.unique(gt)[1:]  # Retrieve all the unique colors in the mask

    # SAM2 can also accept multiple prompts for multiple objects by providing an array of boxes/points and labels
    total_prompts = []  

    for color in unique_colors:
        # Create a binary mask for the current color
        color_mask = (gt == color).astype(np.uint8) * 255  # Convert to binary mask
    
        # Get the prompt for the current color
        prompts = get_multiobject_prompt(
            color_mask, image, dataset, bbsize, mode, prompt_finder, n_pos, n_neg, plot_prompts
        )
        if prompts is not None:
            # Add the prompt to the list, preserving the color information
            for p in prompts:
                p.color = color
                p.class_label = dataset.color_to_label.get(color, "unknown")
                p.class_id = dataset.color_to_id.get(color, -1)
                total_prompts.append(p)
                
    return total_prompts
        


def get_labelmap_prompt(
    gt,
    image,
    dataset,
    bbsize=0,
    mode="point",
    prompt_finder="random",
    n_pos=1,
    n_neg=0,
    plot_prompts=False,
) -> list[Prompt]:
    """
    Generates a prompt for each object (= connected component) in the ground truth labelmap.
    A labelmap is a multi-channel image where each channel represents a different object.
    The prompt can be a bounding box, a point, or both, depending on the mode specified.
    """
    if gt.ndim != 3:
        raise ValueError(
            f"Ground truth mask must be 3D np array with shape (channels, H, W). Got shape {gt.shape}."
        )

    # Generate a prompt for each channel in the labelmap
    prompts = []
    for channel in range(gt.shape[0]):
        obj_gt = gt[channel, :, :]  # Extract the channel as a 2D mask
        if np.sum(obj_gt) == 0:  # Skip empty channels
            continue

        prompt = get_multicolor_prompt(
            gt=obj_gt,
            image=image,
            dataset=dataset,
            bbsize=bbsize,
            mode=mode,
            prompt_finder=prompt_finder,
            n_pos=n_pos,
            n_neg=n_neg,
            plot_prompts=False,  # Plotting is handled later
        )
        if prompt is not None:
            # Add the prompt to the list, preserving the channel information
            for p in prompt:
                p.class_label = dataset.color_to_label.get(channel + 1, "unknown")
                p.class_id = dataset.color_to_id.get(channel + 1, -1)
                p.color = channel + 1  # Use channel index as color
                p.channel = channel  # Store the channel information
                prompts.append(p)

    # Plot prompts
    if plot_prompts:
        img = image.copy()
        for prompt in prompts:
            if prompt.is_empty():
                continue
            if prompt.box is not None:
                x1, y1, x2, y2 = prompt.box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            if prompt.point is not None:
                for point in prompt.point:
                    cv2.circle(img, tuple(point), 5, (0, 0, 255), -1)

        plt.imshow(img, cmap="gray")
        plt.title("Generated Prompts")
        plt.axis("off")
        plt.show()

    return prompts

def get_multiobject_prompt(
    gt,
    image,
    dataset,
    bbsize=0,
    mode="point",
    prompt_finder="random",
    n_pos=1,
    n_neg=0,
    plot_prompts=False
):
    """
    Generates a prompt for each object (= not connected component) in the ground truth segmentation mask.
    The prompt can be a bounding box, a point, or both, depending on the mode specified.
    """
    unique_colors = np.unique(gt)[1:]  # Retrieve all the unique colors in the mask
    if len(unique_colors) > 1:
        raise ValueError(
            "The ground truth mask must contain only one color, but multiple objects"
        )
    elif len(unique_colors) == 0:
        raise ValueError("The ground truth mask is empty.")
    elif len(unique_colors) > 1:
        raise ValueError("The ground truth mask already contains multiple colors.")
    
    color = unique_colors[0]

    prompts = get_multicolor_prompt(
        utils.convert_multiobject_to_multicolor(gt),
        image,
        dataset,
        bbsize=bbsize,
        mode=mode,
        prompt_finder=prompt_finder,
        n_pos=n_pos,
        n_neg=n_neg,
        plot_prompts=plot_prompts,
    )

    for prompt in prompts:
        prompt.class_label = dataset.color_to_label.get(color, "unknown")
        prompt.class_id = dataset.color_to_id.get(color, -1)
        prompt.color = color

    return prompts


def get_yolo_prompt(image, model):
    """
    Generates a bounding box prompt using a YOLO model.
    The bounding box is obtained from the model's predictions on the input image.

    Parameters:
        image (numpy.ndarray): The input image to perform inference on.
        model (YOLO): The YOLO model to use for inference.

    Returns:
        numpy.ndarray: The bounding box coordinates in the format [x1, y1, x2, y2].
    """
    bboxes = []

    confidence = [
        0.3,
        0.2,
        0.1,
        0.02,
        0.008,
    ]  # Minimum confidence thresholds to identify objects
    iou_thresholds = [
        0.7,
        0.6,
        0.5,
        0.2,
        0.2,
    ]  # Minimum IoU thresholds for non-max suppression
    max_detections = [10, 4, 3, 2, 1]  # Maximum number of objects to detect per image

    # Iterate through the confidence, IoU, and max detections thresholds
    for conf, iou, max_det in zip(confidence, iou_thresholds, max_detections):
        results = model.predict(
            source=image,  # Use the image from the dataset
            conf=conf,  # Confidence threshold
            iou=iou,  # IoU threshold for non-max suppression
            max_det=max_det,  # Maximum number of detections per image
            verbose=False,  # Suppress verbose output
        )
        bboxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        if len(bboxes) > 0:  # If bounding boxes are detected, break the loop
            break

    # Return all bounding boxes as a list of numpy arrays
    boxes = []
    for bbox, class_id in zip(bboxes, classes):
        x1, y1, x2, y2 = map(int, bbox[:4])
        boxes.append(np.array([x1, y1, x2, y2]), class_id)

    # As map statement
    return list(map(lambda box, cls: Prompt(box=box, class_id=cls, class_label=model.names[cls], generated_by="yolo"), boxes))
