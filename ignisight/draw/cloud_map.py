import numpy as np
import scipy.io as scio
import vtk
from vtkmodules.util.numpy_support import numpy_to_vtk


def draw3Dtext(point, text, color, orientation):
    atext = vtk.vtkVectorText()
    atext.SetText(text)
    # 说明一下vtkTransform这里是我的需求是要求文件翻转
    trans = vtk.vtkTransform()
    # trans.RotateX(180)
    # trans.RotateY(180)
    # trans.Translate(0,1,0)
    trans.Scale(30, 30, 30)  # 缩放比
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
    # textActor.GetTextProperty().SetJustificationToCentered()
    textActor.GetProperty().SetColor(color)
    return textActor


if __name__ == "__main__":
    source_data = np.load("data-bin/localtion.npy")
    # 新建 vtkPoints 实例
    points = vtk.vtkPoints()
    # 导入点数据
    points.SetData(numpy_to_vtk(source_data))

    # 新建 vtkPolyData 实例
    polydata = vtk.vtkPolyData()
    # 设置点坐标
    polydata.SetPoints(points)

    width = 384
    height = 288
    data = scio.loadmat("data-bin/tempData.mat")  # 384*288的二维温度矩阵
    tempData = np.array(data["thermalImage"])  # 将matlab数据赋值给python变量
    T_height = np.size(tempData, 0)
    T_width = np.size(tempData, 1)
    scalars = vtk.vtkFloatArray()
    y = 0
    for element in range(287, 0, -1):
        for x in range(width):

            value1 = float(
                tempData[np.min([T_height - 1, element]), np.min([T_width - 1, x])]
            )
            # value1 = tempCorrection.Correction(value1)
            scalars.InsertTuple1(y * width + x, value1)
        y = y + 1
    scalarRange = scalars.GetRange()
    polydata.GetPointData().SetScalars(scalars)
    # 顶点相关的 filter
    vertex = vtk.vtkVertexGlyphFilter()
    vertex.SetInputData(polydata)

    lut = vtk.vtkLookupTable()
    lut.SetHueRange(0.39, 0)  # 色调范围 调色板中的色调值除以256得到的浮点数
    lut.SetNumberOfColors(256)  # 颜色个数
    lut.Build()

    mapper = vtk.vtkDataSetMapper()
    # 关联 filter 输出
    mapper.SetInputConnection(vertex.GetOutputPort())

    mapper.SetScalarRange(scalarRange)
    mapper.ScalarVisibilityOn()
    mapper.SetLookupTable(lut)

    # actor 实例
    actor = vtk.vtkActor()
    # 关联 mapper
    actor.SetMapper(mapper)
    actor.GetProperty().SetPointSize(3)
    actor.GetProperty().SetOpacity(1)  # 设置透明度
    actor.GetProperty().SetColor(255, 0, 0)

    # Create a render window
    render = vtk.vtkRenderer()

    point1 = [
        [-300, 440, 0],
        [-320, 520, 0],
        [-320, 600, 0],
        [-320, 680, 0],
        [-320, 760, 0],
        [-320, 840, 0],
        [-320, 920, 0],
        [-100, 760, -200],
        [-100, 760, 200],
        [-100, 520, 200],
    ]
    Ttext = [
        str(round(tempData[193, 22])),  # 硅碳棒1温度
        str(round(tempData[193, 54])),  # 硅碳棒2温度
        str(round(tempData[193, 78])),  # 硅碳棒3温度
        str(round(tempData[193, 92])),  # 硅碳棒4温度
        str(round(tempData[193, 99])),  # 硅碳棒5温度
        str(round(tempData[193, 104])),  # 硅碳棒6温度
        str(round(tempData[193, 108])),  # 硅碳棒7温度
        str(round(tempData[38, 173])),  # 匣钵表面1温度
        str(round(tempData[169, 173])),  # 匣钵表面2温度
        str(round(tempData[169, 225])),
    ]  # 匣钵表面3温度
    color = [0, 0, 0]  # 黑色
    orientation = [90, 0, 90]
    textActors = []
    for i in range(len(Ttext)):
        textActor = draw3Dtext(point1[i], Ttext[i], color, orientation)
        textActors.append(textActor)
        render.AddActor(textActors[i])

    # Insert Actor
    render.AddActor(actor)
    render.GradientBackgroundOn()  # 开启渐变
    render.SetBackground(0.015, 0.427, 0.764)
    render.SetBackground2(0.03, 0.18, 0.4)
    # 图例
    legendBar = vtk.vtkScalarBarActor()
    legendBar.SetOrientationToVertical()
    legendBar.SetLookupTable(lut)
    legenProp = vtk.vtkTextProperty()
    legenProp.SetColor(1, 1, 1)
    legenProp.SetFontSize(18)
    legenProp.SetFontFamilyToArial()
    legenProp.ItalicOff()
    legenProp.BoldOff()
    legendBar.UnconstrainedFontSizeOn()
    legendBar.SetTitleTextProperty(legenProp)
    legendBar.SetLabelTextProperty(legenProp)
    legendBar.SetLabelFormat("%5.2f")
    legendBar.SetTitle("tempereture")

    render.AddActor(legendBar)
    # Renderer Window
    renderWindows = vtk.vtkRenderWindow()
    renderWindows.AddRenderer(render)
    renderWindows.SetSize(1200, 1200)
    renderWindows.SetWindowName("红外三维温度云图")

    # System Event
    iwin_render = vtk.vtkRenderWindowInteractor()
    iwin_render.SetRenderWindow(renderWindows)

    # Style
    iwin_render.SetInteractorStyle(vtk.vtkInteractorStyleMultiTouchCamera())

    iwin_render.Initialize()
    renderWindows.Render()
    iwin_render.Start()
