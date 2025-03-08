import numpy as np
import scipy.io as scio
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

        # 创建 ColorTransferFunction（类似 Plasma colormap）
        self.lut = vtk.vtkColorTransferFunction()
        self.lut.AddRGBPoint(0, 0.050, 0.029, 0.529)  # 深紫色
        self.lut.AddRGBPoint(250, 0.294, 0.000, 0.730)  # 蓝紫色
        self.lut.AddRGBPoint(500, 0.792, 0.000, 0.508)  # 洋红色
        self.lut.AddRGBPoint(750, 0.976, 0.620, 0.224)  # 橙金色
        self.lut.AddRGBPoint(1000, 0.976, 0.984, 0.000)  # 亮金色

        # 创建数据映射器
        self.mapper = vtk.vtkDataSetMapper()
        self.mapper.SetInputConnection(self.vertex.GetOutputPort())
        self.mapper.SetLookupTable(self.lut)
        self.mapper.SetScalarRange(0, 1000)  # 设置固定温度范围
        self.mapper.ScalarVisibilityOn()

        # 创建点云 Actor
        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self.mapper)
        self.actor.GetProperty().SetPointSize(3)
        self.actor.GetProperty().SetOpacity(1)

        # 构建渲染器及背景色
        self.render = vtk.vtkRenderer()
        self.render.AddActor(self.actor)
        self.render.GradientBackgroundOn()
        self.render.SetBackground(0.015, 0.427, 0.764)
        self.render.SetBackground2(0.03, 0.18, 0.4)

        # 创建标量条图例
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
        self.legendBar.SetTitle("Temperature (°C)")
        self.render.AddActor(self.legendBar)

        # 创建渲染窗口和交互器
        self.renderWindow = vtk.vtkRenderWindow()
        self.renderWindow.AddRenderer(self.render)
        self.renderWindow.SetSize(1200, 1200)
        self.renderWindow.SetWindowName("红外三维温度云图")

        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.renderWindow)
        self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleMultiTouchCamera())

        # 用于模拟温度数据变化的计数器
        self.update_counter = 0
        # 存储文本标注 Actor 的列表（方便更新时移除旧的）
        self.text_actors = []

    def update_temperature(self, tempData: np.ndarray):
        """
        接收传入的温度 numpy 数组，并更新点云的标量数据及文本标注。
        :param tempData: 温度数据矩阵（二维 numpy 数组）
        """

        # 翻转温度矩阵（使得点云与温度矩阵对应），再展平成一维数组
        reshaped_temp = self.tempData[::-1].flatten().astype(np.float32)

        # **优化点**：直接使用 numpy_to_vtk 进行高效转换
        scalars = numpy_to_vtk(reshaped_temp, deep=True)

        # scalarRange = scalars.GetRange()
        self.polydata.GetPointData().SetScalars(scalars)
        self.mapper.SetScalarRange(0, 1000)  # 设置固定范围

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

        # 先移除之前添加的文本标注（如果存在）
        for actor in self.text_actors:
            self.render.RemoveActor(actor)
        self.text_actors.clear()

        # 添加新的文本 Actor 到渲染器
        for pos, text in zip(point_positions, Ttext):
            text_actor = self.draw3Dtext(pos, text, color, orientation)
            self.render.AddActor(text_actor)
            self.text_actors.append(text_actor)

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

    def get_temperature_data(self) -> np.ndarray:
        """
        模拟接收新的温度数据。
        实际使用时，可在此方法中添加数据采集逻辑（例如从传感器或网络接收数据）。
        此处模拟一个 (250, 224) 的温度矩阵，数据会随时间变化。
        """
        data = scio.loadmat(
            "tmp-workspace/匣钵区域温度校正/第一组(240901-240902)/温度矩阵/202409011713.mat"
        )
        tempData = np.array(data["thermalImage"])
        # 对温度数据进行边缘填充，保持数据尺寸一致
        tempData = np.pad(tempData, ((0, 0), (0, 2)), "edge")
        base_temp = 30.0
        # 利用正弦波制造周期性温度波动
        variation = 100 * np.sin(self.update_counter / 10.0)
        noise = np.random.randn(288, 384)
        self.tempData = base_temp + variation + noise + tempData
        self.update_counter += 1
        return self.tempData

    def timer_callback(self, obj, event):
        """
        定时器回调函数，每次触发时获取新数据，更新温度点云及文本标注，并刷新渲染窗口。
        """
        new_temp = self.get_temperature_data()
        self.update_temperature(new_temp)
        self.renderWindow.Render()

    def run(self):
        """
        启动渲染窗口和交互，同时设置定时器实现实时数据更新。
        """
        self.interactor.Initialize()
        self.renderWindow.Render()
        # 添加定时器，每隔 1000 毫秒触发一次回调
        self.interactor.AddObserver("TimerEvent", self.timer_callback)
        self.interactor.CreateRepeatingTimer(10)
        self.interactor.Start()


# 示例用法：启动实时温度云图（这里不再从文件加载 mat 数据，而是使用定时器模拟实时数据更新）
if __name__ == "__main__":
    visualizer = TemperatureCloudVisualizer()
    visualizer.run()
