import vtk


def draw3Dtext(point, text, color, orientation):
    atext = vtk.vtkVectorText()
    atext.SetText(text)
    # 说明一下vtkTransform这里是我的需求是要求文件翻转
    trans = vtk.vtkTransform()
    # trans.RotateX(180)
    # trans.RotateY(180)
    # trans.Translate(0,1,0)
    trans.Scale(30, 30, 30)     # 缩放比
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
