import os
import sys

from loadingLbl import LoadingLabel
from script import TextPredictor

# Get the absolute path of the current script file
script_path = os.path.abspath(__file__)

# Get the root directory by going up one level from the script directory
project_root = os.path.dirname(os.path.dirname(script_path))

sys.path.insert(0, project_root)
sys.path.insert(0, os.getcwd())  # Add the current directory as well

from PyQt5.QtWidgets import QMainWindow, QPushButton, QApplication, QLabel, QVBoxLayout, QLineEdit, \
    QWidget
from PyQt5.QtCore import Qt, QCoreApplication, QThread, pyqtSignal
from PyQt5.QtGui import QFont

QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
QCoreApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)  # HighDPI support

QApplication.setFont(QFont('Arial', 12))


class Thread(QThread):
    generateFinished = pyqtSignal(str)

    def __init__(self, text, pred: TextPredictor):
        super(Thread, self).__init__()
        self.__text = text
        self.__pred = pred

    def run(self):
        try:
            result = 'Positive' if self.__pred.predict_text(self.__text) else 'Negative'
            self.generateFinished.emit(result)
        except Exception as e:
            raise Exception(e)


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.__initVal()
        self.__initUi()

    def __initVal(self):
        model_path = 'result.pth'
        self.__pred = TextPredictor(model_path)

    def __initUi(self):
        self.setWindowTitle('PyTorch Text Classification')

        self.__textLineEdit = QLineEdit()
        self.__textLineEdit.setPlaceholderText('Input the Text...')
        self.__textLineEdit.textChanged.connect(self.__textChanged)

        self.__runBtn = QPushButton('Run')
        self.__runBtn.setEnabled(False)
        self.__runBtn.clicked.connect(self.__run)

        self.__waitLbl = LoadingLabel()
        self.__waitLbl.setVisible(False)

        self.__resultLbl = QLabel()
        self.__resultLbl.setAlignment(Qt.AlignCenter)
        self.__resultLbl.setVisible(False)

        lay = QVBoxLayout()
        lay.addWidget(self.__textLineEdit)
        lay.addWidget(self.__runBtn)
        lay.addWidget(self.__waitLbl)
        lay.addWidget(self.__resultLbl)

        mainWidget = QWidget()
        mainWidget.setLayout(lay)

        self.setCentralWidget(mainWidget)

    def __textChanged(self, url):
        self.__runBtn.setEnabled(url.strip() != '')

    def __run(self):
        text = self.__textLineEdit.text()
        self.__t = Thread(text, self.__pred)
        self.__t.started.connect(self.__started)
        self.__t.generateFinished.connect(self.__generateFinished)
        self.__t.finished.connect(self.__finished)
        self.__t.start()

    def __started(self):
        self.__waitLbl.setVisible(True)
        self.__runBtn.setEnabled(False)

    def __generateFinished(self, result):
        self.__resultLbl.setText(result)

    def __finished(self):
        self.__waitLbl.setVisible(False)
        self.__resultLbl.setVisible(True)
        self.__runBtn.setEnabled(True)


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())