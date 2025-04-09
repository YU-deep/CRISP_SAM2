import vtk
from visualization_config import *


def read_volume(file_name):
    if file_name.endswith(".nii.gz"):
        reader = vtk.vtkNIFTIImageReader()
    elif file_name.endswith(".nrrd"):
        reader = vtk.vtkNrrdReader()
    reader.SetFileNameSliceOffset(1)
    reader.SetDataByteOrderToBigEndian()
    reader.SetFileName(file_name)
    reader.Update()
    return reader


def create_mask_extractor(reader):
    """
    Given the output from mask (vtkNIFTIImageReader) extract it into 3D using
    vtkDiscreteMarchingCubes algorithm (https://www.vtk.org/doc/release/5.0/html/a01331.html).
    This algorithm is specialized for reading segmented volume FLARE22.
    :param mask: AbdomenCT-1k vtkNIFTIImageReader volume containing the mask
    :return: the extracted volume from vtkDiscreteMarchingCubes
    """
    mask_extractor = vtk.vtkDiscreteMarchingCubes()
    mask_extractor.SetInputConnection(reader.GetOutputPort())
    return mask_extractor


def create_smoother(reducer, smooth_factor):
    """
    Reorients some points in the volume to smooth the render edges.
    (https://www.vtk.org/doc/nightly/html/classvtkSmoothPolyDataFilter.html)
    """
    smoother = vtk.vtkSmoothPolyDataFilter()
    smoother.SetInputConnection(reducer.GetOutputPort())
    smoother.SetNumberOfIterations(smooth_factor)
    smoother.BoundarySmoothingOn()
    return smoother


def create_mapper(stripper):
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(stripper.GetOutputPort())
    mapper.ScalarVisibilityOff()
    return mapper


def create_property(opacity=0.9, color=(1.0, 0.0, 0.0)):
    prop = vtk.vtkProperty()
    prop.SetColor(color[0], color[1], color[2])
    prop.SetOpacity(opacity)
    # prop.SetRepresentationToWireframe()
    return prop


def create_actor(mapper, prop):
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.SetProperty(prop)
    return actor


def create_renderer(bg_color):
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(bg_color[0], bg_color[1], bg_color[2])
    return renderer


def create_renderwindow(window_name=APPLICATION_TITLE, window_size=(600, 600)):
    render_window = vtk.vtkRenderWindow()
    render_window.SetWindowName(window_name)
    render_window.SetSize(window_size[0], window_size[1])
    return render_window
