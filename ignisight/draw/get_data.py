# 获取深度图像

import cv2
import vtk
import math
import pandas as pd

from vtkmodules.vtkRenderingCore import vtkInteractorObserver

# filename = r"E:\qibiao\ansys\MyProject\Modeling\SpaceClaim\Box-type furnace modeling\xiangshilu.obj"

filename = "xiangshilu.obj"

reader = vtk.vtkOBJReader()
reader.SetFileName(filename)

mapper = vtk.vtkPolyDataMapper()

mapper.SetInputConnection(reader.GetOutputPort())

actor = vtk.vtkActor()

actor.SetMapper(mapper)


# Create a rendering window and renderer
ren = vtk.vtkRenderer()
# Assign actor to the renderer
ren.AddActor(actor)

# camera = vtk.vtkCamera()
camera = ren.GetActiveCamera()
camera.Azimuth(0)
camera.Elevation(0)
camera.Roll(0)
camera.OrthogonalizeViewUp()
camera.SetPosition(-200, 300, 250)
camera.SetFocalPoint(100, 5000, 250)
# camera.SetPosition(-200, 300, 0)
# camera.SetFocalPoint(100, 5000, 0)


# camera.SetViewUp(-1, 0, 0)
# camera.SetViewAngle(81.8)
camera.SetViewAngle(81.8)

# ren.SetActiveCamera(camera)

renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren)  # ren
width = 384
height = 288
renWin.SetSize(width, height)

# cx = 2303
# cy = 2298
# fx = 107.9
# fy = 85.5
# width = 384
# height = 288
#
# # 将主点转化为归一化的图像坐标系
# wcx = -2*(cx - width/2) / width
# wcy = 2*(cy - height/2) / height
# camera.SetWindowCenter(wcx, wcy)

# 将焦距转化为视场角
# view_angle = (2.0 * math.atan2(height/2.0, fy)) * 180/math.pi
# camera.SetViewAngle(view_angle)

renWin.Render()
wti = vtk.vtkWindowToImageFilter()
wti.SetInput(renWin)
jpegWriter = vtk.vtkJPEGWriter()
filename1 = "虚拟图像.jpeg"
jpegWriter.SetFileName(filename1)
jpegWriter.SetInputConnection(wti.GetOutputPort())
jpegWriter.Write()
# renWin.Finalize()

data = vtk.vtkFloatArray()
depthImage = [[] for i in range(height)]
renWin.GetZbufferData(0, 0, width - 1, height - 1, data)
wPos = [0, 0, 0, 0]
wPosList = [[] for i in range(height * width)]
for y in range(height):
    for x in range(width):
        depth = data.GetValue(y*width+x)
        if depth == 1:
            depthImage[y].append(0)
        else:
            vtkInteractorObserver.ComputeDisplayToWorld(ren, x, y, depth, wPos)
            wPosList[y * width + x].append(wPos[0])
            wPosList[y * width + x].append(wPos[1])
            wPosList[y * width + x].append(wPos[2])
            z = wPos[1]-300
            depthImage[y].append(z)

with open("坐标点2.txt", 'w') as f:
    for i in range(height * width):
        f.write(str(wPosList[i][0]) + "  ")
        f.write(str(wPosList[i][1]) + "  ")
        f.write(str(wPosList[i][2]) + "\n")
# for i in range(height):
#     depthImage[i] = list(reversed(depthImage[i]))
depthImage = list(reversed(depthImage))

column = range(width)
img_dep_384x288 = pd.DataFrame(columns=column, data=depthImage)
img_dep_384x288.to_csv('img_dep_384x288.csv')

# Create a renderwindowinteractor
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)

# Enable user interface interactor
iren.Initialize()
renWin.Render()
iren.Start()


