import numpy as np
import cv2

class SeamCarver:
    def __init__(self, filename, out_height, out_width):
        # initialize parameter
        self.filename = filename
        self.out_height = out_height
        self.out_width = out_width

        # read in image and store as np.float64 format
        self.in_image = cv2.imread(filename).astype(np.float64)
        self.in_height, self.in_width = self.in_image.shape[: 2]

        # keep tracking resulting image
        self.out_image = np.copy(self.in_image)

        # kernel for forward energy map calculation
        self.kernel_x = np.array([[0., 0., 0.], [-1., 0., 1.], [0., 0., 0.]], dtype=np.float64)
        self.kernel_y_left = np.array([[0., 0., 0.], [0., 0., 1.], [0., -1., 0.]], dtype=np.float64)
        self.kernel_y_right = np.array([[0., 0., 0.], [1., 0., 0.], [0., -1., 0.]], dtype=np.float64)

        self.seams_carving()

    def seams_carving(self):
        """
        :return:

        We first process seam insertion or removal in vertical direction then followed by horizontal direction.

        The algorithm is written for seam processing in vertical direction (column), so image is rotated 90 degree
        counter-clockwise for seam processing in horizontal direction (row)
        """

        # calculate number of rows and columns needed to be inserted or removed
        delta_row, delta_col = int(self.out_height - self.in_height), int(self.out_width - self.in_width)

        # remove column
        if delta_col < 0:
            self.col_image = self.set_seams_tracking()
            self.seams_removal(delta_col * -1)
            self.col_image = self.visualize_removed_seams(self.col_image)

        elif delta_col > 0:
            print(f"The new height is larger than the orginal height ({self.out_height} > {self.in_height})")

        # remove row
        if delta_row < 0:
            self.out_image = self.rotate_image(self.out_image, 1)
            self.row_image = self.set_seams_tracking()
            self.seams_removal(delta_row * -1)
            self.row_image = self.visualize_removed_seams(self.row_image)
            self.row_image = self.rotate_image(self.row_image, 0)

            self.out_image = self.rotate_image(self.out_image, 0)
            
        elif delta_row > 0:
            print(f"The new width is larger than the orginal width ({self.out_width} > {self.in_width})")

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
        b_energy = self.manual_energy_func(b)
        g_energy = self.manual_energy_func(g)
        r_energy = self.manual_energy_func(r)
        return b_energy + g_energy + r_energy
            
    def cumulative_map_forward(self, energy_map):
        matrix_x = self.calc_neighbor_matrix(self.kernel_x)
        matrix_y_left = self.calc_neighbor_matrix(self.kernel_y_left)
        matrix_y_right = self.calc_neighbor_matrix(self.kernel_y_right)

        m, n = energy_map.shape
        output = np.copy(energy_map)
        for row in range(1, m):
            for col in range(n):
                if col == 0:
                    e_right = output[row - 1, col + 1] + matrix_x[row - 1, col + 1] + matrix_y_right[row - 1, col + 1]
                    e_up = output[row - 1, col] + matrix_x[row - 1, col]
                    output[row, col] = energy_map[row, col] + min(e_right, e_up)
                elif col == n - 1:
                    e_left = output[row - 1, col - 1] + matrix_x[row - 1, col - 1] + matrix_y_left[row - 1, col - 1]
                    e_up = output[row - 1, col] + matrix_x[row - 1, col]
                    output[row, col] = energy_map[row, col] + min(e_left, e_up)
                else:
                    e_left = output[row - 1, col - 1] + matrix_x[row - 1, col - 1] + matrix_y_left[row - 1, col - 1]
                    e_right = output[row - 1, col + 1] + matrix_x[row - 1, col + 1] + matrix_y_right[row - 1, col + 1]
                    e_up = output[row - 1, col] + matrix_x[row - 1, col]
                    output[row, col] = energy_map[row, col] + min(e_left, e_right, e_up)
        return output


    def calc_neighbor_matrix(self, kernel):
        b, g, r = cv2.split(self.out_image)
        output = np.absolute(cv2.filter2D(b, -1, kernel=kernel)) + \
                 np.absolute(cv2.filter2D(g, -1, kernel=kernel)) + \
                 np.absolute(cv2.filter2D(r, -1, kernel=kernel))
        return output


    def find_seam(self, cumulative_map):
        m, n = cumulative_map.shape
        output = np.zeros((m,), dtype=np.uint32)
        output[-1] = np.argmin(cumulative_map[-1])
        for row in range(m - 2, -1, -1):
            prv_x = output[row + 1]
            if prv_x == 0:
                output[row] = np.argmin(cumulative_map[row, : 2])
            else:
                output[row] = np.argmin(cumulative_map[row, prv_x - 1: min(prv_x + 2, n - 1)]) + prv_x - 1
        return output


    def delete_seam(self, seam_idx):
        m, n, c = self.out_image.shape
        mask = np.ones((m, n), dtype=bool)
        mask[np.arange(m), seam_idx] = False  # Mark seam pixels as False
        self.out_image = self.out_image[mask].reshape(m, n - 1, c)

    def rotate_image(self, image, ccw):
        m, n, ch = image.shape
        output = np.zeros((n, m, ch))
        if ccw:
            image_flip = np.fliplr(image)
            for c in range(ch):
                for row in range(m):
                    output[:, row, c] = image_flip[row, :, c]
        else:
            for c in range(ch):
                for row in range(m):
                    output[:, m - 1 - row, c] = image[row, :, c]
        return output


    def rotate_mask(self, mask, ccw):
        m, n = mask.shape
        output = np.zeros((n, m))
        if ccw > 0:
            image_flip = np.fliplr(mask)
            for row in range(m):
                output[:, row] = image_flip[row, : ]
        else:
            for row in range(m):
                output[:, m - 1 - row] = mask[row, : ]
        return output


    def delete_seam_on_mask(self, seam_idx):
        m, n = self.mask.shape
        output = np.zeros((m, n - 1))
        for row in range(m):
            col = seam_idx[row]
            output[row, : ] = np.delete(self.mask[row, : ], [col])
        self.mask = np.copy(output)

    def save_result(self, filename):
        if getattr(self, 'col_image', None) is not None and self.col_image.size > 0:
            cv2.imwrite(filename[:-4] + "_vertical_annotated.jpg", self.col_image.astype(np.uint8))

        if getattr(self, 'row_image', None) is not None and self.row_image.size > 0:
            cv2.imwrite(filename[:-4] + "_horizontal_annotated.jpg", self.row_image.astype(np.uint8))

        cv2.imwrite(filename, self.out_image.astype(np.uint8))



