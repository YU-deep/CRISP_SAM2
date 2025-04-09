import os.path

import SimpleITK as sitk

from visualization_config import *
from utils import *


def mhd_to_nii(file_path: str):
    itk_image = sitk.ReadImage(file_path)
    save_path = file_path.replace(".mhd", ".nii.gz")
    out_arr = sitk.GetArrayFromImage(itk_image)
    out = sitk.GetImageFromArray(out_arr)
    sitk.WriteImage(out, save_path)
    return save_path


if __name__ == '__main__':
    # configs
    show_axes = False
    show_outline = False
    generate_outline_face = False
    take_snapshot = True
    SMOOTH_FACTOR = 400
    ROTATE_X = 270
    ROTATE_Y = 0
    ROTATE_Z = 0
    model_name = 'good'
    dataset_name = 'AMOS22/'
    MASK_COLORS = get_mask_colors(dataset_name)
    MASK_OPACITY = get_mask_opacity(dataset_name)
    file_name = '../input/AMOS22/amos_0409_GT.nii.gz.seg.nrrd'
    if file_name.endswith(".mhd"):
        file_name = mhd_to_nii(file_name)
        output_name = file_name.split('/')[3].split('.')[12]
    else:
        output_name = file_name.split('/')[3].split('.')[0]
    output_path = '../output/' + dataset_name + output_name
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    print(f"file path : {file_name}")
    print(f"output path : {output_path}")
    rotate_config = '_' + str(SMOOTH_FACTOR) + '_' + str(ROTATE_X) + '_' + str(ROTATE_Y) + '_' + str(ROTATE_Z)
    snapshot_filename = output_path + '/' + model_name + rotate_config + '.png'

    # reader
    reader = read_volume(file_name)

    # transform
    mask_transform = vtk.vtkTransform()
    mask_transform.PostMultiply()
    mask_transform.Scale(SCALE)  # scale
    mask_transform.RotateX(ROTATE_X)  # rotate
    mask_transform.RotateY(ROTATE_Y)
    mask_transform.RotateZ(ROTATE_Z)

    # renderer and render window
    renderer = create_renderer(bg_color=RENDERER_BG_COLOR)
    render_window = create_renderwindow()
    render_window.AddRenderer(renderer)
    # render_window.SetSize(100, 100)
    # renderer.SetViewport(0.1, 0.1, 0.9, 0.9)
    # mapper and actors for segmentation results
    n_labels = int(reader.GetOutput().GetScalarRange()[1])
    print(n_labels)

    for idx in range(n_labels):
        extracter = create_mask_extractor(reader)  # extracter
        extracter.SetValue(0, idx + 1)
        smoother = create_smoother(extracter, SMOOTH_FACTOR)  # smoother
        mapper = create_mapper(stripper=smoother)
        prop = create_property(opacity=MASK_OPACITY[idx], color=MASK_COLORS[idx])  # property
        actor = create_actor(mapper=mapper, prop=prop)  # actor
        actor.SetUserTransform(mask_transform)
        renderer.AddActor(actor)

    # outline of the whole image
    if show_outline:
        outline = vtk.vtkOutlineFilter()  # show outline
        outline.SetInputConnection(reader.GetOutputPort())
        if generate_outline_face:  # show surface of the outline
            outline.GenerateFacesOn()
        extracter = create_mask_extractor(reader)
        mapper = create_mapper(stripper=outline)
        prop = create_property(opacity=OUTLINE_OPACITY, color=OUTLINE_COLOR)
        actor = create_actor(mapper=mapper, prop=prop)
        actor.SetUserTransform(mask_transform)
        renderer.AddActor(actor)

    # show axes for better visualization
    if show_axes:
        axes_actor = vtk.vtkAxesActor()
        axes_actor.SetTotalLength(TOTAL_LENGTH[0], TOTAL_LENGTH[1], TOTAL_LENGTH[2])  # set axes length
        axes_actor.SetScale(5, 5, 5)
        renderer.AddActor(axes_actor)

    # start render
    render_window.Render()

    # screenshot
    if take_snapshot:
        w2if = vtk.vtkWindowToImageFilter()
        w2if.SetInput(render_window)
        w2if.SetInputBufferTypeToRGB()
        w2if.ReadFrontBufferOff()
        w2if.Update()

        writer = vtk.vtkPNGWriter()
        writer.SetFileName(snapshot_filename)
        writer.SetInputConnection(w2if.GetOutputPort())
        writer.Write()

    # # interactor
    # interactor = vtk.vtkRenderWindowInteractor()
    # interactor.SetRenderWindow(render_window)
    # interactor.Initialize()
    # interactor.Start()
