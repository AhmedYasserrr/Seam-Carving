import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor

class SeamCarver:
    def __init__(self, filename, out_height, out_width):
        # initialize parameters
        self.filename = filename
        self.out_height = out_height
        self.out_width = out_width

        # read in image and store as np.float64 format
        self.in_image = cv2.imread(filename).astype(np.float64)
        self.in_height, self.in_width = self.in_image.shape[:2]

        # keep tracking resulting image
        self.out_image = np.copy(self.in_image)

        # kernels for forward energy map calculation
        self.kernel_x = np.array([[0., 0., 0.],
                                  [-1., 0., 1.],
                                  [0., 0., 0.]], dtype=np.float64)
        self.kernel_y_left = np.array([[0., 0., 0.],
                                       [0., 0., 1.],
                                       [0., -1., 0.]], dtype=np.float64)
        self.kernel_y_right = np.array([[0., 0., 0.],
                                        [1., 0., 0.],
                                        [0., -1., 0.]], dtype=np.float64)

        self.seams_carving()

    def seams_carving(self):
        """
        Process seam insertion or removal in vertical direction and then horizontal.
        Horizontal processing is done by rotating the image.
        """
        delta_row = int(self.out_height - self.in_height)
        delta_col = int(self.out_width - self.in_width)

        # Remove columns if needed
        if delta_col < 0:
            self.col_image = self.set_seams_tracking()
            self.seams_removal(-delta_col)
            self.col_image = self.visualize_removed_seams(self.col_image)
        elif delta_col > 0:
            print(f"The new width is larger than the original width ({self.out_width} > {self.in_width})")

        # Remove rows if needed: rotate image, process, and rotate back
        if delta_row < 0:
            self.out_image = self.rotate_image(self.out_image, ccw=True)
            self.row_image = self.set_seams_tracking()
            self.seams_removal(-delta_row)
            self.row_image = self.visualize_removed_seams(self.row_image)
            self.row_image = self.rotate_image(self.row_image, ccw=False)
            self.out_image = self.rotate_image(self.out_image, ccw=False)
        elif delta_row > 0:
            print(f"The new height is larger than the original height ({self.out_height} > {self.in_height})")

    def set_seams_tracking(self):
        self.h, self.w, _ = self.out_image.shape
        self.index_map = np.tile(np.arange(self.w), (self.h, 1))
        self.seams = []
        return np.copy(self.out_image)

    def seams_removal(self, num_pixel):
        for _ in range(num_pixel):
            energy_map = self.calc_energy_map()
            cumulative_map = self.cumulative_map_forward(energy_map)
            seam_idx = self.find_seam(cumulative_map)

            # Store seam indices for visualization
            self.seams.append(self.index_map[np.arange(self.h), seam_idx].copy())

            self.delete_seam(seam_idx)

            # Update index map by shifting columns left after seam removal
            for row in range(self.h):
                self.index_map[row, seam_idx[row]:-1] = self.index_map[row, seam_idx[row] + 1:]

    def visualize_removed_seams(self, org_image):
        for seam in self.seams:
            org_image[np.arange(self.h), seam] = [0, 0, 255]  # Mark seams in red
        return org_image

    def manual_energy_func(self, image):
        Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
        Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
        Gx = cv2.filter2D(image, -1, Kx)  # Gradient in x-direction
        Gy = cv2.filter2D(image, -1, Ky)  # Gradient in y-direction
        return np.abs(Gx) + np.abs(Gy)

    def calc_energy_map(self):
        b, g, r = cv2.split(self.out_image)
        # Process each channel concurrently
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.manual_energy_func, channel) for channel in (b, g, r)]
            energies = [future.result() for future in futures]
        return sum(energies)

    def calc_neighbor_matrix(self, kernel):
        b, g, r = cv2.split(self.out_image)
        # Compute filtered outputs for each channel concurrently
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(lambda ch: np.absolute(cv2.filter2D(ch, -1, kernel=kernel)), channel) 
                       for channel in (b, g, r)]
            results = [future.result() for future in futures]
        return sum(results)

    def cumulative_map_forward(self, energy_map):
        # Compute neighbor matrices concurrently for the three kernels
        with ThreadPoolExecutor() as executor:
            future_mx = executor.submit(self.calc_neighbor_matrix, self.kernel_x)
            future_my_left = executor.submit(self.calc_neighbor_matrix, self.kernel_y_left)
            future_my_right = executor.submit(self.calc_neighbor_matrix, self.kernel_y_right)
            matrix_x = future_mx.result()
            matrix_y_left = future_my_left.result()
            matrix_y_right = future_my_right.result()

        m, n = energy_map.shape
        output = np.copy(energy_map)
        # Process each row sequentially (each row depends on the previous)
        for row in range(1, m):
            # Left boundary
            e_up = output[row - 1, 0] + matrix_x[row - 1, 0]
            e_right = output[row - 1, 1] + matrix_x[row - 1, 1] + matrix_y_right[row - 1, 1]
            output[row, 0] = energy_map[row, 0] + min(e_up, e_right)
            # Right boundary
            e_up = output[row - 1, n - 1] + matrix_x[row - 1, n - 1]
            e_left = output[row - 1, n - 2] + matrix_x[row - 1, n - 2] + matrix_y_left[row - 1, n - 2]
            output[row, n - 1] = energy_map[row, n - 1] + min(e_up, e_left)
            # Middle columns: vectorized over the row
            up = output[row - 1, :]
            e_up_full = up + matrix_x[row - 1, :]
            # For left neighbors: shift and pad with infinity at index 0
            e_left = np.empty(n)
            e_left[0] = np.inf
            e_left[1:] = output[row - 1, :-1] + matrix_x[row - 1, :-1] + matrix_y_left[row - 1, :-1]
            # For right neighbors: shift and pad with infinity at last index
            e_right = np.empty(n)
            e_right[-1] = np.inf
            e_right[:-1] = output[row - 1, 1:] + matrix_x[row - 1, 1:] + matrix_y_right[row - 1, 1:]
            # Compute minimum cost for middle columns
            output[row, :] = energy_map[row, :] + np.minimum(np.minimum(e_left, e_up_full), e_right)
        return output

    def find_seam(self, cumulative_map):
        m, n = cumulative_map.shape
        seam = np.zeros(m, dtype=np.uint32)
        # Start at the bottom row with the minimum cumulative energy
        seam[-1] = np.argmin(cumulative_map[-1])
        # Backtrack to find the seam path
        for row in range(m - 2, -1, -1):
            prev_x = seam[row + 1]
            if prev_x == 0:
                idx = np.argmin(cumulative_map[row, :2])
                seam[row] = idx
            elif prev_x == n - 1:
                idx = np.argmin(cumulative_map[row, n - 2:])
                seam[row] = idx + n - 2
            else:
                idx = np.argmin(cumulative_map[row, prev_x - 1:prev_x + 2])
                seam[row] = idx + prev_x - 1
        return seam

    def delete_seam(self, seam_idx):
        m, n, c = self.out_image.shape
        mask = np.ones((m, n), dtype=bool)
        mask[np.arange(m), seam_idx] = False  # Mark seam pixels as False
        self.out_image = self.out_image[mask].reshape(m, n - 1, c)

    def rotate_image(self, image, ccw=True):
        # Use numpy's built-in rotation for efficient vectorized rotation
        return np.rot90(image, k=1 if ccw else -1)

    def rotate_mask(self, mask, ccw=True):
        # Similarly, rotate mask using np.rot90
        return np.rot90(mask, k=1 if ccw else -1)

    def delete_seam_on_mask(self, seam_idx):
        m, n = self.mask.shape
        output = np.zeros((m, n - 1))
        for row in range(m):
            col = seam_idx[row]
            output[row, :] = np.delete(self.mask[row, :], col)
        self.mask = np.copy(output)

    def save_result(self, filename):
        if getattr(self, 'col_image', None) is not None and self.col_image.size > 0:
            cv2.imwrite(filename[:-4] + "_vertical_annotated.jpg", self.col_image.astype(np.uint8))
        if getattr(self, 'row_image', None) is not None and self.row_image.size > 0:
            cv2.imwrite(filename[:-4] + "_horizontal_annotated.jpg", self.row_image.astype(np.uint8))
        cv2.imwrite(filename, self.out_image.astype(np.uint8))
