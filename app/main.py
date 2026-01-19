import sys
from PySide6.QtWidgets import (QApplication, QMainWindow, QMessageBox, 
                               QProgressDialog, QFileDialog, QDialog, 
                               QLabel, QVBoxLayout)
from PySide6.QtCore import Qt, QDate
from Ui_main import Ui_MainWindow

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

import cv2
import numpy as np
from astropy.io.fits import open as fitsopen
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from torch import device
from torchvision import transforms
from Model import AlexNet, VGG, Inception3, ResNet

from warnings import filterwarnings
filterwarnings("ignore")

from Worker import predictWorker, loadModelWorker

class MainWindow(QMainWindow):
    def __init__(self, saveFolder="./temp"):
        super().__init__()

        # 加载UI设计
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self._uiInit() # UI初始化
        self._saveFolderInit(saveFolder) # 保存路径初始化
        self._magInit() # 磁场数据初始化
        self._modelInit() # 模型初始化  
        self._predictInit() # 预测初始化

    def _uiInit(self):
        """UI初始化"""
        # 初始页面
        self.ui.pages.setCurrentIndex(0)
        self.ui.introButton.setChecked(True)
        self.ui.modeTabWidget.setCurrentIndex(4)
        self.ui.modelComboBox.setCurrentIndex(-1)
        self.ui.dateEdit.setDate(QDate.currentDate())

        # 侧边栏按钮
        self.sidebarButtons = [self.ui.introButton, self.ui.modelButton, self.ui.predictButton]
        self.sidebarButtons = [self.ui.introButton, self.ui.modelButton, self.ui.predictButton]
        self.ui.introButton.clicked.connect(lambda: self._switchPage(0))
        self.ui.modelButton.clicked.connect(lambda: self._switchPage(1))
        self.ui.predictButton.clicked.connect(lambda: self._switchPage(2))

        # matplotlib画布
        self.figure = Figure(figsize=(16, 8), dpi=100)  # 调整画布大小
        self.canvas = FigureCanvas(self.figure)  # 创建画布
        self.toolbar = NavigationToolbar(self.canvas, self)  # 创建工具栏
        self.ui.plotVerticalLayout.addWidget(self.toolbar)  # 添加工具栏到布局
        self.ui.plotVerticalLayout.addWidget(self.canvas)  # 添加画布到布局

        # 获取磁场数据
        self.ui.dateEdit.setMinimumDate(QDate(2010, 5, 1))
        self.ui.dateEdit.setMaximumDate(QDate.currentDate())
        self.ui.getMagButton.clicked.connect(lambda: self.getMag())
        # 选择磁场数据
        self.ui.selectMagButton.clicked.connect(lambda: self.selectMag())
        # 选择模型
        self.ui.modelComboBox.currentTextChanged.connect(lambda: self.loadModel())  

        self.progress = None # 进度条
        self.waitDialog = None # 等待对话框

    def _switchPage(self, index):
        """切换页面并更新按钮状态"""
        self.ui.pages.setCurrentIndex(index)
        for i, btn in enumerate(self.sidebarButtons):
            btn.setChecked(i == index)

    def _saveFolderInit(self, saveFolder):
        """初始化保存路径"""
        self.saveFolder = saveFolder
        os.makedirs(self.saveFolder, exist_ok=True)
        os.makedirs(os.path.join(self.saveFolder, "fits"), exist_ok=True)
        os.makedirs(os.path.join(self.saveFolder, "result"), exist_ok=True)
        os.makedirs(os.path.join(self.saveFolder, "picture"), exist_ok=True)

    def _magInit(self):
        """初始化磁场数据"""
        self.currentMagPath = None
        self.currentMagName = None
        self.currentMag = None
    
    def _modelInit(self):
        """初始化模型"""
        self.currentModel = None
        self.modelList = ["AlexNet", "VGG", "Inception3", "ResNet"]
        # self.modelSavePath = "./assets/checkpoint"
        # self.modelSavePath = "_internal/checkpoint"
        self.modelSavePath = "checkpoint"
        self.modelConfig = {
            "AlexNet":{
                "model": AlexNet,
                "checkpoint": os.path.join(self.modelSavePath, "AlexNet_best.pth")
            },
            "VGG":{
                "model": VGG,
                "checkpoint": os.path.join(self.modelSavePath, "VGG_best.pth")
            },
            "Inception3":{
                "model": Inception3,
                "checkpoint": os.path.join(self.modelSavePath, "Inception3_best.pth")
            },
            "ResNet":{
                "model": ResNet,
                "checkpoint": os.path.join(self.modelSavePath, "ResNet_best.pth")
            }
        }

    def _predictInit(self):
        self.predictResult = None
        self.device = device("cpu")
        self.trans = transforms.Compose([transforms.ToTensor()]) # 数据预处理
        self.ui.truepredictButton.clicked.connect(lambda: self.magPredict()) # 预测
   
    def _loadModelWorker(self):
        modelConfig = self.modelConfig[self.modelList[self.ui.modelComboBox.currentIndex()]]
        self.loadModelWorker = loadModelWorker(
            modelConfig["model"], modelConfig["checkpoint"]
        )
        self.loadModelWorker.finished.connect(lambda result: self._loadModelFinished(result)) # 加载完成

    def _loadModelStart(self):
        self.loadModelWorker.start() # 开始加载

    def _stopLoadModelWorker(self):
        if self.loadModelWorker and self.loadModelWorker.isRunning():
            self.loadModelWorker.requestInterruption()
            self.loadModelWorker.quit()
            self.loadModelWorker.wait(500)

    def _loadModelFinished(self, message):
        if self.waitDialog: self.waitDialog.destroy()
        if isinstance(message, Exception): 
            self.ui.modelComboBox.setCurrentIndex(-1)
            self._messageBox("错误", f"模型加载失败: \n{str(message)}").exec()
            self.currentModel = None
        else:
            self.currentModel = message
            self._stopLoadModelWorker()
            self._messageBox("成功", "模型加载完成！").exec()

    def _predictWorker(self):
        self.predictWorker = predictWorker(
            self.currentMag, self.currentModel, self.trans, self.device,
            self.figure, self.canvas, self.currentMagName, self.saveFolder
        )
        self.predictWorker.finished.connect(lambda result: self._predictFinished(result)) # 预测完成

    def _predictStart(self):
        self.ui.truepredictButton.setEnabled(False)
        self.predictWorker.start() 

    def _stopPredictWorker(self):
        if self.predictWorker and self.predictWorker.isRunning():
            self.predictWorker.requestInterruption()
            self.predictWorker.quit()
            self.predictWorker.wait(500)

    def _predictFinished(self, message):
        self.ui.truepredictButton.setEnabled(True)
        if self.waitDialog: 
            self.waitDialog.destroy()
        if isinstance(message, Exception):
            self._messageBox("错误", f"模型预测失败:\n{str(message)}").exec()
        else:
            [probNF, probC, probM, probX] = message
            self.ui.predictResult.setText(f"预测结果: 无耀斑:{probNF:.5f}, C级耀斑:{probC:.5f}, M级耀斑:{probM:.5f}, X级耀斑:{probX:.5f}")
            with open(os.path.join(self.saveFolder, "result", f"{self.currentMagName}.txt"), 'a', encoding='utf-8') as f:
                f.write(f"{self.currentModel._get_name()} 预测结果: 无耀斑:{probNF:.5f}, C级耀斑:{probC:.5f}, M级耀斑:{probM:.5f}, X级耀斑:{probX:.5f}\n")
            self._stopPredictWorker()

    def _messageBox(self, title, text):
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.NoIcon) # 设置为无图表 ==> 静音
        msg.setWindowTitle(title)
        msg.setText(text)
        msg.addButton("确定", QMessageBox.AcceptRole) # 添加中文按钮
        return msg
    
    def _progressDialog(self, labelText, cancelButtonText, windowTitle):
        progress = QProgressDialog(labelText, cancelButtonText, 0, 100, self)  # 创建进度条
        progress.setWindowTitle(windowTitle)
        progress.setModal(True)  # 设置为模态窗口
        progress.setMinimumDuration(10)  # 设置最小持续时间
        progress.setMaximum(100)  # 设置最大值
        return progress

    def _waitDialog(self, text, windowTitle):
        waitDialog = QDialog(self)
        waitDialog.setWindowTitle(windowTitle)
        waitDialog.setModal(True)  # 设置为模态窗口
        waitDialog.setWindowFlags(waitDialog.windowFlags() & ~Qt.WindowCloseButtonHint) # 禁用关闭按钮
        layout = QVBoxLayout(waitDialog)
        layout.addWidget(QLabel(text))
        waitDialog.setLayout(layout)
        waitDialog.setFixedSize(200, 100)  # 设置固定大小
        return waitDialog

    def getMag(self, baseUrl="https://jsoc1.stanford.edu/data/hmi/fits/"):
        """获取今日磁场数据"""
        date = self.ui.dateEdit.date()
        year = date.year(); month = date.month(); day = date.day()
        saveFolder = os.path.join(self.saveFolder, "fits", f"{year:04d}", f"{month:02d}", f"{day:02d}")
        os.makedirs(saveFolder, exist_ok=True)
        magPage = urljoin(baseUrl, f"{year:04d}/{month:02d}/{day:02d}/")
        try:
            magPageResponse = requests.get(magPage)
            soup = BeautifulSoup(magPageResponse.content, 'html.parser')
            hrefList = [href['href'] for href in soup.find_all(href=re.compile(r'\.fits$'))]

            # 检查数据是否已存在
            if all(os.path.exists(savePath) for savePath in [os.path.join(saveFolder, href) for href in hrefList]):
                self._messageBox("提示", "数据已存在, 无需下载.").exec()
                return
            
            for href in hrefList:
                savePath = os.path.join(saveFolder, href)
                if os.path.exists(savePath): continue

                self.progress = self._progressDialog(f"正在下载{href}", "取消", "下载进度")
                self.progress.show()
                QApplication.processEvents()

                url = urljoin(magPage, href)
                with requests.get(url, stream=True) as response:
                    downloaded = 0
                    chunckSize = 1024  # 1KB
                    total_size = int(response.headers.get('Content-Length', 0))
                    with open(savePath, 'wb') as f:
                        for chunk in response.iter_content(chunckSize):
                            if chunk:
                                f.write(chunk)
                                downloaded += chunckSize
                                self.progress.setValue(int(downloaded / total_size * 100))
                                QApplication.processEvents()
                                if self.progress.wasCanceled():
                                    f.close()
                                    os.remove(savePath)
                                    self.progress.destroy()
                                    self._messageBox("提示", "下载已取消.").exec()
                                    return
                
                self.progress.destroy()
            
            self._messageBox("成功", "下载成功！").exec()

        except requests.ConnectionError as e:
            if self.progress: self.progress.destroy()
            self._messageBox("错误", f"网络连接失败：\n{str(e)}").exec()
        
        except requests.ConnectTimeout as e:
            if self.progress: self.progress.destroy()
            self._messageBox("错误", f"网络连接超时：\n{str(e)}").exec()

        except Exception as e:
            if self.progress: self.progress.destroy()
            self._messageBox("错误", f"意外错误：\n{str(e)}").exec()

    def selectMag(self):
        """选择磁场数据"""
        if self.currentMagPath is None:
            openPath = os.path.join(self.saveFolder, "fits")
        else:
            openPath = os.path.dirname(self.currentMagPath)

        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "选择磁场数据", 
            openPath, 
            "FITS Files (*.fits)"
            )

        if file_path:
            self.currentMagPath = file_path
            self.currentMagName = os.path.basename(self.currentMagPath)
            self.ui.magPath.setText(self.currentMagName)
            self.loadMag()
            self.magVisualize()

    def loadMag(self):
        """加载磁场数据"""
        try:
            hdul = fitsopen(self.currentMagPath)
            data = hdul[1].data.astype(np.float32)
            preprocessData = self._magPreprocess(data)
            self.currentMag = np.stack([preprocessData]*3, axis=-1)
            hdul.close()
        except Exception as e:
            self._messageBox("错误", f"加载磁场数据失败：\n{str(e)}").exec()

    def _magPreprocess(self, data):
        """预处理磁场数据"""
        h, w = data.shape
        left = 0; right = w-1
        while left < w and np.all(np.isnan(data[:, left])): left += 1
        while right >= 0 and np.all(np.isnan(data[:, right])): right -= 1
        top = 0; bottom = h-1
        while top < h and np.all(np.isnan(data[top, :])): top += 1
        while bottom >= 0 and np.all(np.isnan(data[bottom, :])): bottom -= 1
        cropped = np.nan_to_num(data[top:bottom+1, left:right+1])
        resized = cv2.resize(cropped, (512, 512), interpolation=cv2.INTER_LINEAR)
        return resized

    def _resetPlot(self):
        """重置画布"""
        self.figure.clear()  # 清空画布
        self.canvas.draw()  # 画布显示

    def magVisualize(self):
        """磁场数据可视化"""
        try:
            self._resetPlot()  # 重置画布
            ax = self.figure.add_subplot(111)  # 添加子图
            ax.imshow(self.currentMag, cmap='gray')  # 绘制图像
            ax.axis('off')  # 关闭坐标轴
            ax.set_title(f'{self.currentMagName}')  # 设置标题
            self.figure.tight_layout()  # 调整布局
            savePath = os.path.join(self.saveFolder, "picture", f"{self.currentMagName}.png")
            if not os.path.exists(savePath):
                self.figure.savefig(savePath, dpi=300, bbox_inches='tight', pad_inches=0)
            self.canvas.draw()  # 画布显示
        except Exception as e:
            self._messageBox("错误", f"磁场数据可视化失败：\n{str(e)}").exec()

    def loadModel(self):
        """加载模型"""
        modelIndex = self.ui.modelComboBox.currentIndex()

        if modelIndex == -1: return

        self.waitDialog = self._waitDialog("正在加载模型...", "加载模型")
        self.waitDialog.show()
        QApplication.processEvents()

        try:
            self._loadModelWorker()
            self._loadModelStart()
        except Exception as e:
            self.waitDialog.destroy()
            self._messageBox("错误", f"模型加载失败：\n{str(e)}").exec()           

    def magPredict(self):
        if self.currentMag is None:
            self._messageBox("警告", "请选择磁场数据").exec()
            return
        
        if self.currentModel is None:
            self._messageBox("警告", "请选择模型").exec()
            return
        
        self.waitDialog = self._waitDialog("正在预测...", "预测")
        self.waitDialog.show()
        QApplication.processEvents()

        try:
            self._predictWorker()
            self._predictStart()
        except Exception as e:
            self.waitDialog.destroy()
            self._messageBox("错误", f"意外错误：\n{str(e)}").exec()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()