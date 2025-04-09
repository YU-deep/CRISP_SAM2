"""
GUI config
"""
# main window, only for Qt GUI
APPLICATION_TITLE = "3D Segmentation Visualizer"
MIN_IMG_WINDOW_WIDTH = 200

"""
VTK config
"""
COLOR_CONFIG = {
    "liver": (220 / 256, 0 / 256, 0 / 256),
    "spleen": (0 / 256, 220 / 256, 0 / 256),
    "stomach": (0 / 256, 220 / 256, 220 / 256),
    "gallbladder": (220 / 256, 0 / 256, 220 / 256),
    "esophagus": (225 / 256, 210 / 256, 190 / 256),
    "pancreas": (0 / 256, 0 / 256, 250 / 256),
    "duodenum": (200 / 256, 130 / 256, 60 / 256),
    "aorta": (0 / 256, 120 / 256, 120 / 256),
    "bladder": (40 / 256, 140 / 256, 90 / 256),
    "inferior vena cava": (220 / 256, 190 / 256, 150 / 256),
    "left kidney": (40 / 256, 60 / 256, 80 / 256),
    "right kidney": (220 / 256, 220 / 256, 0 / 256),
    "left adrenal gland": (90 / 256, 20 / 256, 110 / 256),
    "right adrenal gland": (200 / 256, 60 / 256, 90 / 256),
    "left femur": (255 / 256, 230 / 256, 225 / 256),
    "right femur": (110 / 256, 90 / 256, 200 / 256),
    "left lung": (100 / 256, 40 / 256, 140 / 256),
    "right lung": (60 / 256, 100 / 256, 200 / 256),
    "default": (1, 1, 1)
}


# mask config
def get_mask_colors(dataset_name: str):
    if "word" in dataset_name.lower():
        return [
            # # WORD
            COLOR_CONFIG["liver"],
            COLOR_CONFIG["spleen"],
            COLOR_CONFIG["left kidney"],
            COLOR_CONFIG["right kidney"],
            COLOR_CONFIG["stomach"],
            COLOR_CONFIG["gallbladder"],
            COLOR_CONFIG["esophagus"],
            COLOR_CONFIG["pancreas"],
            COLOR_CONFIG["duodenum"],
            COLOR_CONFIG["default"],
            COLOR_CONFIG["default"],
            COLOR_CONFIG["default"],
            COLOR_CONFIG["default"],
            COLOR_CONFIG["bladder"],
            COLOR_CONFIG["left femur"],
            COLOR_CONFIG["right femur"],
        ]
    elif "flare" in dataset_name.lower():
        return [
            COLOR_CONFIG["liver"],
            COLOR_CONFIG["right kidney"],
            COLOR_CONFIG["spleen"],
            COLOR_CONFIG["pancreas"],
            COLOR_CONFIG["aorta"],
            COLOR_CONFIG["inferior vena cava"],
            COLOR_CONFIG["right adrenal gland"],
            COLOR_CONFIG["left adrenal gland"],
            COLOR_CONFIG["gallbladder"],
            COLOR_CONFIG["esophagus"],
            COLOR_CONFIG["stomach"],
            COLOR_CONFIG["duodenum"],
            COLOR_CONFIG["left kidney"],
        ]
    elif "abdomen" in dataset_name.lower():
        return [
            COLOR_CONFIG["liver"],
            COLOR_CONFIG["default"],
            COLOR_CONFIG["spleen"],
            COLOR_CONFIG["pancreas"],
            COLOR_CONFIG["left kidney"],
            COLOR_CONFIG["right kidney"],
        ]
    elif "amos" in dataset_name.lower():
        return [
            COLOR_CONFIG["spleen"],
            COLOR_CONFIG["right kidney"],
            COLOR_CONFIG["left kidney"],
            COLOR_CONFIG["gallbladder"],
            COLOR_CONFIG["esophagus"],
            COLOR_CONFIG["liver"],
            COLOR_CONFIG["stomach"],
            COLOR_CONFIG["aorta"],
            COLOR_CONFIG["inferior vena cava"],
            COLOR_CONFIG["pancreas"],
            COLOR_CONFIG["right adrenal gland"],
            COLOR_CONFIG["left adrenal gland"],
            COLOR_CONFIG["duodenum"],
            COLOR_CONFIG["bladder"],
            COLOR_CONFIG["default"],
        ]
    elif "luna" in dataset_name.lower():
        return [
            COLOR_CONFIG["default"],
            COLOR_CONFIG["default"],
            COLOR_CONFIG["right lung"],
            COLOR_CONFIG["left lung"],
            COLOR_CONFIG["default"],
        ]
    elif "spleen" in dataset_name.lower():
        return [
            COLOR_CONFIG["spleen"],
        ]
    elif "pancreas" in dataset_name.lower():
        return [
            COLOR_CONFIG["pancreas"],
            COLOR_CONFIG["default"],
        ]
    else:
        return []


def get_mask_opacity(dataset_name: str):
    if "word" in dataset_name.lower():
        return [
            # # WORD
            1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1
        ]
    elif "flare" in dataset_name.lower():
        return [
            # # FLARE22
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
        ]
    elif "abdomen" in dataset_name.lower():
        return [
            1, 1, 1, 1, 1, 1
        ]
    elif "amos" in dataset_name.lower():
        return [
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0
        ]
    elif "luna" in dataset_name.lower():
        return [
            0, 0, 1, 1, 0
        ]
    elif "spleen" in dataset_name.lower():
        return [
            1
        ]
    elif "pancreas" in dataset_name.lower():
        return [
            1, 0
        ]
    else:
        return []


SMOOTH_FACTOR = 400
MAX_LABEL_LENGTH = 10

# renderer
COMPARE = True
RENDERER_BG_COLOR = (1., 1., 1.)

# outline config
SHOW_OUTLINE = True
OUTLINE_COLOR = (0, 1, 1)
OUTLINE_OPACITY = 0.2

# transform config
ROTATE_X = 270
ROTATE_Y = 180
ROTATE_Z = 0
SCALE = (100, 100, 100)
"""
You should set the rotate config for the best effect of visualization.
"""
# axes config
SHOW_AXES = True
TOTAL_LENGTH = (50, 50, 50)

# # mask config
# MASK_COLORS = [
#     # (245/256, 0/256,   61/256),   # spleen
#
#     # # AbdomenCT-1k
#     (255 / 256, 51 / 256, 204 / 256),  # liver
#     (61 / 256, 245 / 256, 0 / 256),  # kidney
#     (245 / 256, 0 / 256, 61 / 256),  # spleen
#     (209 / 256, 163 / 256, 117 / 256),  # pancreas
#     (61 / 256, 177 / 256, 255 / 256),  # l kidney
#     (61 / 256, 245 / 256, 0 / 256),  # r kidney
#
#     # # WORD
#     # (255/256, 51/256,  204/256),  # liver
#     # (245/256, 0/256,   61/256),   # spleen
#     # (61/256,  177/256, 255/256),  # l kidney
#     # (61/256,  245/256, 0/256),    # r kidney
#     # (255/256, 255/256, 204/256),  # stomach
#     # (255/256, 255/256, 0/256),    # gallbladder
#     # (0/256,   255/256, 255/256),  # esophagus
#     # (209/256, 163/256, 117/256),  # pancreas
#     # (255/256, 153/256, 204/256),  # duodenum
#     # (153/256, 128/256, 0/256),    # colon
#     # (256/256, 256/256, 256/256),  # intestine (invisible)
#     # (255/256, 128/256, 0/256),    # adrenal gland
#     # (51/256,  0/256,   0/256),    # rectum
#     # (255/256, 102/256, 0/256),    # bladder
#     # (128/256, 128/256, 128/256),  # l femur
#     # (128/256, 128/256, 128/256),  # r femur
#
#     # # LUNA
#
#     # # BTCV
#     # (245/256, 0/256,   61/256),   # spleen
#     # (61/256,  245/256, 0/256),    # r kidney
#     # (61/256,  177/256, 255/256),  # l kidney
#     # (255/256, 255/256, 0/256),    # gallbladder
#     # (0/256,   255/256, 255/256),  # esophagus
#     # (255/256, 51/256,  204/256),  # liver
#     # (255/256, 255/256, 204/256),  # stomach
#     # (0/256,   0/256,   255/256),  # aorta
#     # (194/256, 71/256,  71/256),   # inferior_vena_cava (postcava)
#     # (133/256, 71/256,  194/256),  # portal_vein_and_splenic_vein
#     # (209/256, 163/256, 117/256),  # pancreas
#     # (255/256, 128/256, 0/256),    # r adrenal gland
#     # (255/256, 128/256, 0/256),    # l adrenal gland
#
# ]
