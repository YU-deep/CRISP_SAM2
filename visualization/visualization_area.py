from PIL import Image


def zoom_in_one_area(image_path, output_path, box, zoom_factor):
    image = Image.open(image_path)
    left, upper, right, lower = box
    assert left * upper * right * lower <= 1
    left = left * image.size[0]
    upper = upper * image.size[1]
    right = right * image.size[0]
    lower = lower * image.size[1]
    # print(image.size)
    cropped_image = image.crop((left, upper, right, lower))
    new_size = (int(cropped_image.width * zoom_factor), int(cropped_image.height * zoom_factor))
    zoomed_image = cropped_image.resize(new_size, Image.LANCZOS)
    zoomed_image.save(output_path)


image_path = '../output/...'
output_path = '../output/...'


box = (0.6, 0.6, 0.8, 0.8)   # (x_upper_left, y_upper_left, x_lower_right, y_lower_right) should be [0, 1]
zoom_factor = 2

zoom_in_one_area(image_path, output_path, box, zoom_factor)
