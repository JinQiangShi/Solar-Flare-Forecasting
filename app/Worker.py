import os
from numpy import ogrid, where, zeros, float32
from PySide6.QtCore import QThread, Signal
from torch import load, device, softmax
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

class loadModelWorker(QThread):
    finished = Signal(object) # 完成信号

    def __init__(self, model, checkpoint):
        super().__init__()
        self.model = model
        self.checkpoint = checkpoint

    def run(self):
        try:
            model = self.model()
            model.load_state_dict(load(self.checkpoint, map_location=device("cpu")))
            # print(model._get_name())
            self.finished.emit(model)
            
        except Exception as error:
            self.finished.emit(error)

class predictWorker(QThread):
    finished =  Signal(object) # 完成信号
    
    def __init__(self, mag, model, trans, device, figure, canvas, title, saveFolder):
        super().__init__()
        self.mag = mag
        self.model = model
        self.trans = trans
        self.device = device
        self.figure = figure
        self.canvas = canvas
        self.temperature = 1
        self.title = title
        self.saveFolder = saveFolder

        self.modelName = model._get_name()
        if self.modelName == "AlexNet":
            self.target_layers = [model.features[-1]]
        elif self.modelName == "VGG":
            self.target_layers = [model.features[-1]]
        elif self.modelName == "Inception3":
            self.target_layers = [model.Mixed_7c]
            self.temperature = 0.2
        elif self.modelName == "ResNet":
            self.target_layers = [model.layer4[-1]]
        
        self.cam = GradCAM(model=self.model, target_layers=self.target_layers)
    
    def run(self):
        self.model.eval()
        self.model.to(self.device)
        input = self.trans(self.mag).unsqueeze(0)
        input = input.to(self.device)

        try:
            grayscaleCam = []
            for cls in [2,3]:
                target = [ClassifierOutputTarget(cls)]
                grayscaleCam.append(self.cam(input_tensor=input, targets=target)[0, :])
            self.camVisualization(grayscaleCam)
            output = self.cam.outputs
            self.finished.emit(softmax(output * self.temperature, dim=1).detach().numpy()[0])
        
        except Exception as error:
            self.finished.emit(error)
    
    def circleMask(self, size=512):
        center = size // 2
        radius = size // 2
        y, x = ogrid[:size, :size]
        mask = ((x - center) ** 2 + (y - center) ** 2 <= radius ** 2)
        rgbaMask = zeros((*mask.shape, 4), dtype=float32)
        rgbaMask[..., :3] = 0.0  # 黑色背景
        rgbaMask[..., 3] = where(mask, 0.0, 1.0)  # 掩码区域透明
        return rgbaMask

    def _resetPlot(self):
        """重置画布"""
        self.figure.clear()  # 清空画布
        self.canvas.draw()  # 画布显示

    def camVisualization(self, grayscaleCam):
        mask = self.circleMask()
        self._resetPlot()
        ax = self.figure.add_subplot(111)
        ax.imshow(self.mag, cmap="gray")
        ax.imshow(grayscaleCam[0]+grayscaleCam[1], cmap="jet", alpha=0.5)
        ax.imshow(mask)
        ax.axis("off")
        ax.set_title(f"{self.title} {self.modelName} Grad-CAM")
        self.canvas.draw()  # 画布显示

        savePath = os.path.join(self.saveFolder, "picture",  f"{self.title} {self.modelName} Grad-CAM.png")
        if not os.path.exists(savePath):
            self.figure.savefig(savePath, dpi=300, bbox_inches='tight', pad_inches=0)
