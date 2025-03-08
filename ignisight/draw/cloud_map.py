import numpy as np
import vtk
from vtkmodules.util.numpy_support import numpy_to_vtk


class TemperatureCloudVisualizer:
    def __init__(self, point_file="data-bin/localtion.npy"):
        """
        初始化点云渲染管线，仅依赖于点数据。
        :param point_file: 存储点坐标的 .npy 文件路径
        """
        # 加载点数据并构建 vtkPoints
        self.source_data = np.load(point_file)
        self.points = vtk.vtkPoints()
        self.points.SetData(numpy_to_vtk(self.source_data))

        # 构建 vtkPolyData
        self.polydata = vtk.vtkPolyData()
        self.polydata.SetPoints(self.points)

        # 创建 Glyph 过滤器，将点数据转换为可渲染的形式
        self.vertex = vtk.vtkVertexGlyphFilter()
        self.vertex.SetInputData(self.polydata)

        # 设置 LookupTable，用于后续标量数据映射
        self.lut = vtk.vtkLookupTable()
        self.lut.SetHueRange(0.39, 0)
        self.lut.SetNumberOfColors(256)
        self.lut.Build()

        # 创建数据映射器
        self.mapper = vtk.vtkDataSetMapper()
        self.mapper.SetInputConnection(self.vertex.GetOutputPort())
        self.mapper.SetLookupTable(self.lut)
        self.mapper.ScalarVisibilityOn()

        # 创建点云 Actor
        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self.mapper)
        self.actor.GetProperty().SetPointSize(3)
        self.actor.GetProperty().SetOpacity(1)
        self.actor.GetProperty().SetColor(255, 0, 0)

        # 构建渲染器及背景色
        self.render = vtk.vtkRenderer()
        self.render.AddActor(self.actor)
        self.render.GradientBackgroundOn()
        self.render.SetBackground(0.015, 0.427, 0.764)
        self.render.SetBackground2(0.03, 0.18, 0.4)

        # 构建标量条图例
        self.legendBar = vtk.vtkScalarBarActor()
        self.legendBar.SetOrientationToVertical()
        self.legendBar.SetLookupTable(self.lut)
        legenProp = vtk.vtkTextProperty()
        legenProp.SetColor(1, 1, 1)
        legenProp.SetFontSize(18)
        legenProp.SetFontFamilyToArial()
        self.legendBar.SetTitleTextProperty(legenProp)
        self.legendBar.SetLabelTextProperty(legenProp)
        self.legendBar.SetLabelFormat("%5.2f")
        self.legendBar.SetTitle("temperature")
        self.render.AddActor(self.legendBar)

        # 创建渲染窗口和交互器
        self.renderWindow = vtk.vtkRenderWindow()
        self.renderWindow.AddRenderer(self.render)
        self.renderWindow.SetSize(1200, 1200)
        self.renderWindow.SetWindowName("红外三维温度云图")

        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.renderWindow)
        self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleMultiTouchCamera())

    def update_temperature(self, tempData: np.ndarray):
        """
        接收传入的温度 numpy 数组，并更新点云的标量数据及文本标注。
        :param tempData: 温度数据矩阵（二维 numpy 数组）
        """
        # 对温度数据进行边缘填充，保持数据尺寸一致
        self.tempData = np.pad(tempData, ((0, 0), (0, 2)), "edge")

        # 翻转温度矩阵（使得点云与温度矩阵对应），再展平成一维数组
        reshaped_temp = self.tempData[::-1].flatten().astype(np.float32)

        # **优化点**：直接使用 numpy_to_vtk 进行高效转换
        scalars = numpy_to_vtk(reshaped_temp, deep=True)

        scalarRange = scalars.GetRange()
        self.polydata.GetPointData().SetScalars(scalars)
        self.mapper.SetScalarRange(scalarRange)

        # 同时添加文本标注
        self.add_text_annotations()

    def add_text_annotations(self):
        """
        根据传入的温度数据添加文本标注，示例中固定使用部分数据索引和预设位置。
        如果需要更灵活的设置，可以对方法进行扩展。
        """
        if not hasattr(self, "tempData"):
            return  # 未调用 update_temperature

        # 示例中的文本坐标和数据索引
        point_positions = [
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
            str(round(self.tempData[193, 22])),
            str(round(self.tempData[193, 54])),
            str(round(self.tempData[193, 78])),
            str(round(self.tempData[193, 92])),
            str(round(self.tempData[193, 99])),
            str(round(self.tempData[193, 104])),
            str(round(self.tempData[193, 108])),
            str(round(self.tempData[38, 173])),
            str(round(self.tempData[169, 173])),
            str(round(self.tempData[169, 225])),
        ]
        color = [0, 0, 0]  # 黑色文本
        orientation = [90, 0, 90]  # 旋转角度

        # 添加文本 Actor 到渲染器
        for pos, text in zip(point_positions, Ttext):
            text_actor = self.draw3Dtext(pos, text, color, orientation)
            self.render.AddActor(text_actor)

    @staticmethod
    def draw3Dtext(
        point: list, text: str, color: list, orientation: list
    ) -> vtk.vtkFollower:
        atext = vtk.vtkVectorText()
        atext.SetText(text)

        trans = vtk.vtkTransform()
        trans.Scale(30, 30, 30)  # 缩放

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

    def run(self):
        """
        启动渲染窗口和交互。
        """
        self.interactor.Initialize()
        self.renderWindow.Render()
        self.interactor.Start()


# 示例用法
if __name__ == "__main__":
    import scipy.io as scio

    # 从外部加载温度数据（例如 mat 文件中的 thermalImage）
    data = scio.loadmat(
        "tmp-workspace/匣钵区域温度校正/第一组(240901-240902)/温度矩阵/202409011713.mat"
    )
    tempData = np.array(data["thermalImage"])

    # 创建可视化对象，并传入温度数据进行更新渲染
    visualizer = TemperatureCloudVisualizer()
    visualizer.update_temperature(tempData)
    visualizer.run()
