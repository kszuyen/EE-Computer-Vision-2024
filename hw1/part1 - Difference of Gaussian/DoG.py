import numpy as np
import cv2


class Difference_of_Gaussian(object):

    def __init__(self, threshold):
        self.threshold = threshold
        self.sigma = 2 ** (1 / 4)
        self.num_octaves = 2
        self.num_DoG_images_per_octave = 4
        self.num_guassian_images_per_octave = self.num_DoG_images_per_octave + 1

    def get_dog_images(self, image):
        # Step 1: Filter images with different sigma values (5 images per octave, 2 octave in total)
        # - Function: cv2.GaussianBlur (kernel = (0, 0), sigma = self.sigma**___)
        gaussian_images = []

        for _ in range(self.num_octaves):
            gaussian_images_in_octave = []
            gaussian_images_in_octave.append(image)
            # first image in octave already has the correct blur
            for i in range(self.num_DoG_images_per_octave):
                gaussian_images_in_octave.append(
                    cv2.GaussianBlur(
                        image,
                        (0, 0),
                        sigmaX=self.sigma ** (i + 1),
                    )
                )
            gaussian_images.append(gaussian_images_in_octave)
            octave_base = gaussian_images_in_octave[-1]
            image = cv2.resize(
                octave_base,
                (int(octave_base.shape[1] / 2), int(octave_base.shape[0] / 2)),
                interpolation=cv2.INTER_NEAREST,
            )

        # Step 2: Subtract 2 neighbor images to get DoG images (4 images per octave, 2 octave in total)
        # - Function: cv2.subtract(second_image, first_image)

        dog_images = []

        for gaussian_images_in_octave in gaussian_images:
            dog_images_in_octave = []
            for first_image, second_image in zip(gaussian_images_in_octave[:], gaussian_images_in_octave[1:]):
                dog_images_in_octave.append(
                    cv2.subtract(second_image, first_image)
                )  # ordinary subtraction will not work because the images are unsigned integers
            dog_images.append(dog_images_in_octave)

        return dog_images

    def get_keypoints(self, image):
        ### TODO ####

        dog_images = self.get_dog_images(image)

        # Step 3: Thresholding the value and Find local extremum (local maximun and local minimum)
        #         Keep local extremum as a keypoint

        def isPixelAnExtremum(first_subimage, second_subimage, third_subimage, threshold):
            """Return True if the center element of the 3x3x3 input array is strictly greater than or less than all its neighbors, False otherwise"""
            center_pixel_value = second_subimage[1, 1]
            if abs(center_pixel_value) > threshold:
                if center_pixel_value > 0:
                    return (
                        np.all(center_pixel_value >= first_subimage)
                        and np.all(center_pixel_value >= third_subimage)
                        and np.all(center_pixel_value >= second_subimage[0, :])
                        and np.all(center_pixel_value >= second_subimage[2, :])
                        and center_pixel_value >= second_subimage[1, 0]
                        and center_pixel_value >= second_subimage[1, 2]
                    )
                else:
                    return (
                        np.all(center_pixel_value <= first_subimage)
                        and np.all(center_pixel_value <= third_subimage)
                        and np.all(center_pixel_value <= second_subimage[0, :])
                        and np.all(center_pixel_value <= second_subimage[2, :])
                        and center_pixel_value <= second_subimage[1, 0]
                        and center_pixel_value <= second_subimage[1, 2]
                    )
            else:
                return False

        keypoints = []

        for octave_index, dog_images_in_octave in enumerate(dog_images):
            for _, (first_image, second_image, third_image) in enumerate(
                zip(
                    dog_images_in_octave,
                    dog_images_in_octave[1:],
                    dog_images_in_octave[2:],
                )
            ):
                # (i, j) is the center of the 3x3 array
                for i in range(1, first_image.shape[0] - 1):
                    for j in range(1, first_image.shape[1] - 1):
                        if isPixelAnExtremum(
                            first_image[i - 1 : i + 2, j - 1 : j + 2],
                            second_image[i - 1 : i + 2, j - 1 : j + 2],
                            third_image[i - 1 : i + 2, j - 1 : j + 2],
                            self.threshold,
                        ):
                            keypoints.append([i * 2**octave_index, j * 2**octave_index])

        # Step 4: Delete duplicate keypoints
        # - Function: np.unique
        keypoints = np.unique(keypoints, axis=0)

        # sort 2d-point by y, then by x
        keypoints = keypoints[np.lexsort((keypoints[:, 1], keypoints[:, 0]))]
        return keypoints
