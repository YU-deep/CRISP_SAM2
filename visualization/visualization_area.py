from PIL import Image


def zoom_in_one_area(image_path, output_path, box, zoom_factor):
    image = Image.open(image_path)
    left, upper, right, lower = box
    cropped_image = image.crop((left, upper, right, lower))
    new_size = (int(cropped_image.width * zoom_factor), int(cropped_image.height * zoom_factor))
    zoomed_image = cropped_image.resize(new_size, Image.LANCZOS)
    zoomed_image.save(output_path)


image_path = '../output/AMOS22/amos_0409/ctsam3d_400_270_0_0.png'
output_path = '../output/AMOS22/amos_0409_zoom/ctsam3d_3.png'
factor = 1024 / 480
factor = 1
# print(110 * factor)
# print(150 * factor)
# print(280 * factor)
# print(320 * factor)
box = (245 * factor, 410 * factor, 385 * factor, 550 * factor)   # (x_upper_left, y_upper_left, x_lower_right, y_lower_right)
zoom_factor = 2

zoom_in_one_area(image_path, output_path, box, zoom_factor)
