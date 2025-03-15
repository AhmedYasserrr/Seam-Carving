from Parallelized_seam_carving import SeamCarver
import os
import time

if __name__ == '__main__':
    """
    Put image in in/images folder
    Ouput image will be saved to out/images folder with filename_output
    """

    folder_in = 'in'
    folder_out = 'out'
    filename_input = 'image.jpg'
    filename_output = 'image_result.png'
    new_height = 278
    new_width =  600

    input_image = os.path.join(folder_in, "images", filename_input)
    output_image = os.path.join(folder_out, "images", filename_output)

    start_time = time.time()  
    
    obj = SeamCarver(filename_input, new_height, new_width)
    obj.save_result(filename_output)

    end_time = time.time()
    print(f"image resize took {end_time - start_time:.4f} seconds")






