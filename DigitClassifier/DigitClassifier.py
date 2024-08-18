# Warning - Only Works For The Digit Images Of The MNIST Dataset 
#Any Other Class Of Image is Highly Likely To Produce Error In classification



from PyQt5.QtWidgets import QVBoxLayout,QHBoxLayout,QLabel,QFileDialog,QApplication,QPushButton,QSpacerItem,QSizePolicy,QWidget
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap,QIcon,QImage
import sys
import cv2
import numpy as np

class MainAppWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setGeometry(0,0,750,850)
        self.setWindowTitle("MNIST Digit Classifier")
        icon=QIcon('favicon.ico')
        self.setWindowIcon(icon)
        self.setFixedSize(750,850)
        background_label=QLabel(self)
        background_image=QPixmap("Background.png")
        
        background_label.setPixmap(background_image)
        background_label.setScaledContents(True) 

        background_label.setGeometry(self.rect())
        self.window2=None
        self.init_ui()
        

    def init_ui(self):
        layout=QVBoxLayout()
        layout.addSpacerItem(QSpacerItem(10,30,QSizePolicy.Minimum,QSizePolicy.Expanding))
        start=QPushButton("Classify Now")
        start.clicked.connect(self.nextwin)
        start.setStyleSheet('''
                            color:white;
                            margin:3px;
                            font-size:40px;
                            border-radius:20px;
                            background-color:#3E4F5E;
                            padding:3px;
                            ''')
        layout.addWidget(start,alignment=Qt.AlignCenter | Qt.AlignBottom)
        self.setLayout(layout)
    
    def nextwin(self):
        self.window2=SecondaryWindow()
        self.window2.show()
        self.destroy()
    

class SecondaryWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setGeometry(0,0,750,850)
        self.setWindowTitle("MNIST Digit Classifier")
        icon=QIcon('favicon.ico')
        self.setWindowIcon(icon)
        self.setFixedSize(750,850)
        self.newwin=None
        background_label=QLabel(self)
        background_label.setStyleSheet('''
                                       background-color:#0B1721;''')
        background_label.setScaledContents(True) 

        background_label.setGeometry(self.rect())
        self.init_ui()
    
    def init_ui(self):
        self.layout=QVBoxLayout()
        self.open_button=QPushButton("Select An Image")
        self.open_button.clicked.connect(self.openFile)
        self.open_button.setStyleSheet('''
                            color:white;
                            font-size:40px;
                            border-radius:20px;
                            background-color:#3E4F5E;
                            padding:3px;
                            ''')
        self.layout.addWidget(self.open_button)
        self.setLayout(self.layout)
        
    
    def openFile(self):
        opt=QFileDialog.Options()
        file_filter = "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)"
        filepath,file=QFileDialog.getOpenFileName(self,"Open Image File","",file_filter,options=opt)
        if filepath:
            
            file=cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
            file=cv2.resize(file,(28,28),interpolation=cv2.INTER_AREA)
            file=file/255.0
            file=file.reshape(-1,28*28)
            self.classify(file)
            self.display_image(filepath)
    
    def display_image(self,file):
        img=QImage(file)
        img=img.scaled(400,400)
        img_label=QLabel()
        img_label.setPixmap(QPixmap(img))
        img_label.setStyleSheet(    '''    QLabel {
                            font-size:20px;
                            background-color:black;
                            border: 20px solid #16b851;
                            border-radius:20px;
                            color:white;
                                }
                       '''     
        )
        self.layout.addWidget(img_label,alignment=Qt.AlignCenter)
        result_label=QLabel(f"The Digit In The Image is : ")
        result_label.setStyleSheet('''
                                   font-size:30px;
                                   color:white;
                                   font-family:Bahnscrift;
                                   ''')
        self.layout.addWidget(result_label,alignment=Qt.AlignCenter)
        number_label=QLabel(list(str(self.prediction))[1])
        number_label.setStyleSheet('''
                                   font-size:90px;
                                   color:white;
                                   font-family:Bahnscrift;
                                   ''')
        self.layout.addWidget(number_label,alignment=Qt.AlignCenter)
        self.changeButton()

    
    def changeButton(self):
        self.open_button.setText("Classify Again !")
        self.open_button.clicked.disconnect()
        self.open_button.clicked.connect(self.prevWin)
    
    def prevWin(self):
        self.newwin=SecondaryWindow()
        self.newwin.show()
        self.destroy()
  

        

  

    
    def classify(self,file):
        data = np.load(r"model_weights_biases.npz")
        w1 = data['w1']
        b1 = data['b1']
        w2 = data['w2']
        b2 = data['b2']
        w3=data['w3']
        b3=data['b3']  
        model=ClassifyStructure(w1,b1,w2,b2,w3,b3)
        self.prediction=model.predict(file)
        print(self.prediction)



class ClassifyStructure:
    def __init__(self,w1,b1,w2,b2,w3,b3):
        self.w1,self.b1=w1,b1
        self.w2,self.b2=w2,b2
        self.w3,self.b3=w3,b3
    
    def ReLU(self,x):
        return np.maximum(0.01*x,x)
    
    def softmax(self,z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def predict(self,x):
        z1=np.dot(x,self.w1)+self.b1
        o1=self.ReLU(z1)
        z2=np.dot(o1,self.w2)+self.b2
        o2=self.ReLU(z2)
        z3=np.dot(o2,self.w3)+self.b3
        o3=self.softmax(z3)
        return np.argmax(o3,axis=1)
        

       
if __name__=="__main__":
    app=QApplication([])
    window=MainAppWindow()
    window.show()
    sys.exit(app.exec_())