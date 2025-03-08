import numpy as np
import scipy.io as scio
import vtk
from vtkmodules.util.numpy_support import numpy_to_vtk

def draw3Dtext(point: list, text: str, color: list, orientation: list) -> vtk.vtkFollower:
    """
    绘制 3D 文本。
    :param point: 文本的坐标点 [x, y, z]
    :param text: 显示的文本内容
    :param color: 文本颜色 [R, G, B]
    :param orientation: 旋转角度 [X, Y, Z]
    :return: 3D 文字 Actor
    """
    atext = vtk.vtkVectorText()
    atext.SetText(text)
    
    trans = vtk.vtkTransform()
    trans.Scale(30, 30, 30)  # 设置缩放比
    
    tf = vtk.vtkTransformFilter()
    tf.SetTransform(trans)
    tf.SetInputConnection(atext.GetOutputPort())
    
    textMapper = vtk.vtkPolyDataMapper()
    textMapper.SetInputConnection(tf.GetOutputPort())
    
    textActor = vtk.vtkFollower()
    textActor.SetMapper(textMapper)
    textActor.AddPosition(point)
    textActor.RotateX(orientation[0])
    textActor.RotateY(orientation[1])
    textActor.RotateZ(orientation[2])
    textActor.GetProperty().SetColor(color)
    
    return textActor

if __name__ == "__main__":
    # 加载点坐标数据
    source_data = np.load("data-bin/localtion.npy")
    # 新建 vtkPoints 实例
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(source_data))
    # 新建 vtkPolyData 实例，并设置点数据
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    # 读取温度数据
    data = scio.loadmat("data-bin/tempData.mat")
    tempData = np.array(data["thermalImage"])  # 提取温度矩阵
    tempData = np.pad(tempData, ((0, 0), (0, 2)), "edge")
    # 获取温度矩阵的尺寸
    T_height, T_width = tempData.shape
    # 创建存储标量数据的数组
    scalars = vtk.vtkFloatArray()
    # Reshape the temperature data to match our point cloud structure
    # Flip the temperature matrix upside down (using [::-1]) to match the for-loop's reversed order
    reshaped_temp = tempData[::-1].flatten().astype(np.float32)
    # Convert the numpy array to VTK array in one operation
    for i in range(len(reshaped_temp)):
        scalars.InsertTuple1(i, reshaped_temp[i])
    scalarRange = scalars.GetRange()
    polydata.GetPointData().SetScalars(scalars)
    # 创建点的 Glyph 过滤器
    vertex = vtk.vtkVertexGlyphFilter()
    vertex.SetInputData(polydata)
    # 设置颜色映射表
    lut = vtk.vtkLookupTable()
    lut.SetHueRange(0.39, 0)
    lut.SetNumberOfColors(256)
    lut.Build()
    # 映射数据
    mapper = vtk.vtkDataSetMapper()
    mapper.SetInputConnection(vertex.GetOutputPort())
    mapper.SetScalarRange(scalarRange)
    mapper.ScalarVisibilityOn()
    mapper.SetLookupTable(lut)
    # 创建 Actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetPointSize(3)
    actor.GetProperty().SetOpacity(1)
    actor.GetProperty().SetColor(255, 0, 0)
    # 创建渲染器
    render = vtk.vtkRenderer()
    # 定义文本位置和内容
    point1 = [
        [-300, 440, 0], [-320, 520, 0], [-320, 600, 0],
        [-320, 680, 0], [-320, 760, 0], [-320, 840, 0],
        [-320, 920, 0], [-100, 760, -200], [-100, 760, 200],
        [-100, 520, 200]
    ]
    Ttext = [
        str(round(tempData[193, 22])),
        str(round(tempData[193, 54])),
        str(round(tempData[193, 78])),
        str(round(tempData[193, 92])),
        str(round(tempData[193, 99])),
        str(round(tempData[193, 104])),
        str(round(tempData[193, 108])),
        str(round(tempData[38, 173])),
        str(round(tempData[169, 173])),
        str(round(tempData[169, 225]))
    ]
    color = [0, 0, 0]  # 黑色文本
    orientation = [90, 0, 90]
    # textActors = []
    for i in range(len(Ttext)):
        textActor = draw3Dtext(point1[i], Ttext[i], color, orientation)
        # textActors.append(textActor)
        render.AddActor(textActor)
    # 添加主 Actor
    render.AddActor(actor)
    render.GradientBackgroundOn()
    render.SetBackground(0.015, 0.427, 0.764)
    render.SetBackground2(0.03, 0.18, 0.4)
    # 创建图例
    legendBar = vtk.vtkScalarBarActor()
    legendBar.SetOrientationToVertical()
    legendBar.SetLookupTable(lut)
    
    legenProp = vtk.vtkTextProperty()
    legenProp.SetColor(1, 1, 1)
    legenProp.SetFontSize(18)
    legenProp.SetFontFamilyToArial()
    legendBar.SetTitleTextProperty(legenProp)
    legendBar.SetLabelTextProperty(legenProp)
    legendBar.SetLabelFormat("%5.2f")
    legendBar.SetTitle("temperature")
    
    render.AddActor(legendBar)
    
    # 创建渲染窗口
    renderWindows = vtk.vtkRenderWindow()
    renderWindows.AddRenderer(render)
    renderWindows.SetSize(1200, 1200)
    renderWindows.SetWindowName("红外三维温度云图")
    
    # 交互窗口
    iwin_render = vtk.vtkRenderWindowInteractor()
    iwin_render.SetRenderWindow(renderWindows)
    iwin_render.SetInteractorStyle(vtk.vtkInteractorStyleMultiTouchCamera())
    
    iwin_render.Initialize()
    renderWindows.Render()
    iwin_render.Start()
