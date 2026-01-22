from PyQt6.QtWidgets import QMainWindow, QApplication, QVBoxLayout, QFileDialog, \
    QDialog, QDoubleSpinBox, QDialogButtonBox
from PyQt6.QtGui import QStandardItemModel, QStandardItem
from PyQt6.QtCore import QPropertyAnimation, QParallelAnimationGroup, QRect, QSize, QThread, pyqtSignal
from PyQt6.uic import loadUi
#from PyQt5.QtGui import QFileDialog

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import RectangleSelector
from matplotlib.path import Path

import numpy as np
import pandas as pd
import pybaselines as pb
#from scipy.fftpack import fftfreq, irfft, rfft
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import wiener

from sif_tools import sif2array

import sys, os
#os.chdir("C:/Users/adato/OneDrive - Danmarks Tekniske Universitet/Dokumenter/Programs/qtspectra/AndorV2/")

class Canvas(FigureCanvasQTAgg):
    def __init__(self, parent):

        self.fig, self.ax = plt.subplots(figsize=(15,15), dpi=100)       
        super().__init__(self.fig)

        self.setParent(parent)
        """
        Matplotlib script
        """
        a = np.random.random(1024)

        self.ax.set(xlabel = "Raman shift [cm$^{-1}$]", ylabel = "Intensity", title = "Spectrum")
        self.ax.plot(range(0, len(a)), a)        

    def update_plot(self, data, index, smoothedData=[], baseline=[]):
        
        self.ax.clear()

        print(f"Index length : {len(index)}, data : {data}")
        
        # self.ax.plot(index, data.values)
        self.ax.plot(index, data.values)

        if len(smoothedData) > 0:
            self.ax.plot(index, smoothedData)
            print(f"Smoothed data: {smoothedData}")
        elif len(baseline) > 0:

            self.ax.plot(index, baseline)
        else:
            print(f"No smoothing")
            pass
            
        self.ax.set(xlabel = "Raman shift [cm$^{-1}$]", ylabel = "Intensity", title = "Spectrum")
        self.fig.canvas.draw()

    def savePlot(self, filename):
        if filename != "":
            self.fig.savefig(filename)
        else: 
            print("No file selected.")

class HeatmapPlot(FigureCanvasQTAgg):
    def __init__(self, parent):

        #self.fig, self.ax = plt.subplots(ncols=1, nrows=2, figsize=(15,15), dpi=100)
        self.fig = plt.figure(layout="constrained", figsize=(15,15), dpi=100)

        super().__init__(self.fig)

        self.gs = GridSpec(2, 2, figure=self.fig)

        self.scanAx = self.fig.add_subplot(self.gs[0, 0])
        self.heatmapAx = self.fig.add_subplot(self.gs[0, 1])
        self.spectrumAx = self.fig.add_subplot(self.gs[1, :])

        self.setParent(parent)
        self.parent = parent
        """
        Scanned field selection plot matplotlib script
        """
        x = np.random.rand(10)
        y = np.random.rand(10)

        self.scanAx.set(xlabel = "x [μm]", ylabel = "y [μm]", title = "Scanned field")
        #self.scanAx.scatter(x, y)
        self.heatmapAx.set(xlabel = "x [μm]", ylabel = "y [μm]", title = "Spectrum results")

        self.spectrumAx.set(xlabel = "Wavelength [nm]", ylabel = "Intensity", title = "Raman spectrum")
        self.spectrumPlot, = self.spectrumAx.plot([],[], visible=False, color="tab:blue")

        # Setting up rectangle select tool
        self.rectangleSelector = RectangleSelector(self.scanAx, self.line_select_callback,
                                                   minspanx=5, minspany=5,
                                                   spancoords='pixels',
                                                   interactive=True)
        
        # Name of spectrum variable
        self.spectrumName = ""
  
    def line_select_callback(self, eclick, erelease):
        'eclick and erelease are the press and release events'
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        #print("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2))

        x = np.array(self.parent.dfSpac.columns.get_level_values("x"))
        y = np.array(self.parent.dfSpac.columns.get_level_values("y"))

        boolSelectX = np.logical_and(x > min([x1, x2]), x < max([x1, x2]))
        boolSelectY = np.logical_and(y > min([y1, y2]), y < max([y1, y2]))

        boolSelectBoth = np.logical_and(boolSelectX, boolSelectY)

        self.dfSubsetSpac = self.parent.dfSpac.iloc[:,boolSelectBoth]

        subsetX = np.array(self.dfSubsetSpac.columns.get_level_values("x"))
        subsetY = np.array(self.dfSubsetSpac.columns.get_level_values("y"))

        ncols = len(np.unique(subsetX))
        nrows = len(np.unique(subsetY))

        self.subsetX = subsetX.reshape(nrows, ncols)
        self.subsetY = subsetY.reshape(nrows, ncols)
        

        self.subsetZ = np.array(self.dfSubsetSpac.max(axis=0).values).reshape(nrows, ncols)
        
        self.updateHeatmap()
        

    def updateHeatmap(self):
        
        self.heatmapAx.clear()

        self.colormap = self.heatmapAx.pcolormesh(self.subsetX, self.subsetY, self.subsetZ, cmap='RdBu_r', vmin=np.min(self.subsetZ), vmax=np.percentile(self.subsetZ, q=100))

        # Connect the hover event
        self.fig.canvas.mpl_connect('motion_notify_event', self.heatmapHover)

        if "colorbar" in self.__dict__:          
            #self.colormap.set_array(self.subsetZ.ravel())
            #self.colormap.set_clim(vmin=np.min(self.subsetZ), vmax=np.np.percentile(self.subsetZ, q=100))
            #self.colorbar.set_clim(vmin=np.min(self.subsetZ), vmax=np.percentile(self.subsetZ, q=100))
            #self.colorbar.draw_all()
            self.colorbar = self.fig.colorbar(self.colormap, cax=self.colorbar.ax)
        else:
            self.colorbar = self.fig.colorbar(self.colormap, ax=self.heatmapAx)

        self.scanAx.set(xlabel = "x [μm]", ylabel = "y [μm]", title = "Scanned field")
        self.heatmapAx.set(xlabel = "x [μm]", ylabel = "y [μm]", title = "Spectrum results")
        #self.fig.draw_all()

    def heatmapHover(self, event):
        if event.inaxes == self.heatmapAx:  # Check if the event is in the desired subplot (ax1)
            # New data arrived
            self.xCoord = event.xdata
            self.yCoord = event.ydata
            print(f"X: {self.xCoord}, xType : {type(self.xCoord)} Y: {self.yCoord}")
            print(f"event: {event}")

            # Finding the nearest value of X
            xVals = np.array(self.dfSubsetSpac.columns.get_level_values("x"))
            deltaX = np.abs(xVals - self.xCoord)
            boolSelectX = deltaX == np.min(deltaX)

            #print(f"Bool select X : {boolSelectX}")

            yVals = np.array(self.dfSubsetSpac.columns.get_level_values("y"))
            deltaY = np.abs(yVals - self.yCoord)
            boolSelecty = deltaY == np.min(deltaY)

            #print(f"Bool select y : {boolSelecty}")
            boolSelectXY = np.logical_and(boolSelectX, boolSelecty)

            self.spectrumData = self.dfSubsetSpac.iloc[:,boolSelectXY]

            print(f"Spectrum name : {self.spectrumData.columns.get_level_values('Names')[0]}")

            # Checking if this is a new spectrum
            if self.spectrumData.columns.get_level_values('Names')[0] != self.spectrumName:
                self.spectrumName = self.spectrumData.columns.get_level_values('Names')[0]
                self.wavelgth = list(self.spectrumData.index)
                self.values = list(self.spectrumData.iloc[:,0])

                #print(f"Wavelength: {self.wavelgth}, values: {self.values}")

                self.updateSpectrumPlot()

    def updateSpectrumPlot(self):
        print("Updating spectrum plot")
        self.spectrumAx.set_xlim([min(self.wavelgth), max(self.wavelgth)])
        self.spectrumAx.set_ylim([min(self.values), max(self.values)])

        self.spectrumPlot.set_xdata(self.wavelgth)
        self.spectrumPlot.set_ydata(self.values)
        self.spectrumPlot.set_visible(True)

        self.fig.canvas.draw_idle()


    def updateSpacialCoordinates(self):
        self.scanAx.clear()

        x = np.array(self.parent.dfSpac.columns.get_level_values("x"))
        y = np.array(self.parent.dfSpac.columns.get_level_values("y"))
        
        self.scanAx.scatter(x,y)

        minX = min(x) - abs(0.05*min(x))
        minY = min(y) - abs(0.05*min(y))

        maxX = max(x) + abs(0.05*max(x))
        maxY = max(y) + abs(0.05*max(y))
                        
        self.scanAx.set_xlim([minX, maxX])
        self.scanAx.set_ylim([minY, maxY])

        self.rectangleSelector = RectangleSelector(self.scanAx, self.line_select_callback,
                                            minspanx=5, minspany=5,
                                            spancoords='pixels',
                                            interactive=True)

        self.scanAx.set(xlabel = "x [μm]", ylabel = "y [μm]", title = "Scanned field")
        self.heatmapAx.set(xlabel = "x [μm]", ylabel = "y [μm]", title = "Spectrum results")

        self.fig.canvas.draw()
    """
    def update_plot(self, data, index, smoothedData=[], baseline=[]):
        
        self.ax[0].clear()

        print(f"Index length : {len(index)}, data : {data}")
        
        # self.ax.plot(index, data.values)
        self.ax[0].plot(index, data.values)

        if len(smoothedData) > 0:
            self.ax[0].plot(index, smoothedData)
            print(f"Smoothed data: {smoothedData}")
        elif len(baseline) > 0:

            self.ax[0].plot(index, baseline)
        else:
            print(f"No smoothing")
            pass
            
        self.ax[0].set(xlabel = "Raman shift [cm$^{-1}$]", ylabel = "Intensity", title = "Spectrum")
        self.fig.canvas.draw()
    """
    def savePlot(self, filename):
        if filename != "":
            self.fig.savefig(filename)
        else: 
            print("No file selected.")


class main(QMainWindow):
    def __init__(self):
        super().__init__()
        print(os.getcwd())
        #loadUi(r"C:\Users\adato\OneDrive - Danmarks Tekniske Universitet\Dokumenter\Programs\spectra_browser\mainwindow.ui", self)
        loadUi("mainwindow.ui", self)
        
        """
        #Loading spectrum plot
        """
        self.plot = Canvas(self)
        vbox = QVBoxLayout()
        vbox.addWidget(self.plot)
        #vbox.setContentsMargins(0,0,0,0)
        self.figureCanvas.setLayout(vbox)

        """
        #Loading heatmap plot
        """
        self.heatmapPlot = HeatmapPlot(self)
        vbox = QVBoxLayout()
        vbox.addWidget(self.heatmapPlot)
        #vbox.setContentsMargins(0,0,0,0)
        self.heatmapQFrame.setLayout(vbox)

        """
        #Loading file list
        """
        self.model = QStandardItemModel()
        self.listView.setModel(self.model)

        self.listView.selectionModel().currentChanged.connect(self.fileSelectionCallback)

        """
        #Loading menu actions
        """
        self.actionLoad_folder.triggered.connect(self.loadFolder)
        self.actionExit.triggered.connect(self.exitProgram)
        self.filename = ""
        self.savePlotFilePath = ""
        self.actionSave_figure.triggered.connect(self.savePlotCallback)

        self.actionLoad_file.triggered.connect(self.loadFile)

        self.saveDataFilePath = ""
        self.actionSave_data.triggered.connect(self.saveCurrentData)
        
        # Processing tab
        self.actionarPLS.triggered.connect(self.arPLSSmoothing)
        self.actionFurrierFilter.triggered.connect(self.furrierFilter)
        self.actionWienerFilter.triggered.connect(self.weinerFilter)

        """
        #Loading Line tab buttons and actions
        """
        #self.runSmoothingBtn.clicked.connect(self.smoothingCallback)
        #self.smoothingValueCBox.currentTextChanged.connect(lambda: self.frameAnimation("smoothing"))
        # Activating animation
        self.frameAnimation("smoothing")
        self.lastFolder = "C:/Users/adato/OneDrive - Danmarks Tekniske Universitet/Dokumenter/Raman measurements/Raman images"
        #self.lastFolder = "C:/Users/adato/OneDrive - Danmarks Tekniske Universitet/Dokumenter/Raman measurements/test_data/Ecoli/"

        self.loadSpacialCoordinatesBtn.clicked.connect(self.loadSpacialCoordinates)

    def loadSpacialCoordinates(self):
        #print(f"self dict {self.__dict__}")
        if "completeDf" in self.__dict__:
            print(f"[loadSpacialCoordinates] Data is loaded.")
            # Spacial data frame and ordinary data frame are kept separate.
            self.dfSpac = self.completeDf.copy(deep=True)

            xList = []
            yList = []
            zList = []

            for name in self.dfSpac:
                name = name[:-7]
                name = name.split('_')[1:]
                xList.append(float(name[0][2:]))
                yList.append(float(name[1][2:]))
                zList.append(float(name[2][2:]))

            index = pd.MultiIndex.from_frame(pd.DataFrame({"Names": self.dfSpac.columns, "x": xList, "y": yList, "z": zList}))
            self.dfSpac.columns = index

            self.heatmapPlot.updateSpacialCoordinates()

        else:
            print("[loadSpacialCoordinates] Data is not loaded in. Please load data first.")

    def baseline_als(y, lam, p, niter=10):
        L = len(y)
        D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
        w = np.ones(L)
        for i in range(niter):
            W = sparse.spdiags(w, 0, L, L)
            Z = W + lam * D.dot(D.transpose())
            z = spsolve(Z, w*y)
            w = p * (y > z) + (1-p) * (y < z)
        return z
    
    def _ffFilter(self, values, percentile):
        # Furrier transform
        signal = np.fft.fft(values)
        # Absolute values
        signal_abs = np.abs(signal)
        # Percentile
        perc = np.percentile(signal_abs, q = percentile)

        signal[signal_abs < perc] = 0

        filtered_signal = np.real(np.fft.ifft(signal))
        
        return filtered_signal

    def furrierSmoothing(self):
        print(f"[furrierSmoothing] Callback executed.")
        percentile = self.furrierFilter.ffCutoffValue.value()
        # This try/except catches if variable self.completeDf isn't created yet.
        try:
            self.smoothedDf = self.completeDf.copy()

            for entry in self.completeDf:
                smoothed = self._ffFilter(values=self.completeDf[entry].values, percentile=percentile)
                self.smoothedDf[entry] = smoothed
        except BaseException as e:
            print(f"[furrierSmoothing] Exception {e}")
            print(f"[furrierSmoothing] Check if you have loaded data.")
    
        print(f"[furrierSmoothing] selected {self.listView.selectionModel().selectedIndexes()}")
        selectedIndex = self.listView.selectionModel().selectedIndexes()
        # There is always only one item selected, but I can't seem to select anything normally from this stupid list.
        for item in selectedIndex:
            self.fileSelectionCallback(item)

    def weinerSmoothing(self):
        window = self.weinerFilter.windowValue.value()
        if self.weinerFilter.noiseValue.value() == 0:
            noise = None
        else: 
            noise = self.weinerFilter.noiseValue.value()

        try:
            self.smoothedDf = self.completeDf.copy()

            for entry in self.completeDf:
                smoothed = wiener(self.completeDf[entry], mysize=window, noise=noise)
                self.smoothedDf[entry] = smoothed
        except BaseException as e:
            print(f"[weinerSmoothing] Exception {e}")
            print(f"[weinerSmoothing] Check if you have loaded data.")

        
    def arPLSplot(self):
        arPLSLambda = float(self.arPLS.arPLSLambda.text())
        print(f"[arPLScallback] Callback executed. Lambda value : {arPLSLambda}")
        selectedIndex = self.listView.selectionModel().selectedIndexes()
        # There is always only one item selected, but I can't seem to select anything normally from this stupid list.
        for item in selectedIndex:
            baseline_fitter = pb.Baseline(self.completeDf[item.data()].index, check_finite=False)
            background = baseline_fitter.pspline_airpls(self.completeDf[item.data()].values, lam = arPLSLambda)[0]
            # Transforming to raman shift
            # TURNING OFF WAVELENGTH TO SHIFT OPTION
            index = self.wave_to_shift(wavelength = self.completeDf[item.data()].index)
            try:
                self.plot.update_plot(data = self.completeDf[item.data()], index=index, smoothedData=self.smoothedDf[item.data()].values, baseline=background)
            except:
                self.plot.update_plot(data = self.completeDf[item.data()], index=index, baseline=background)

    def arPLSapply(self):
        arPLSLambda = float(self.arPLS.arPLSLambda.text())
        print("[arPLSapply] callback executed.")
        selectedIndex = self.listView.selectionModel().selectedIndexes()
        # There is always only one item selected, but I can't seem to select anything normally from this stupid list.
        for item in selectedIndex:
            baseline_fitter = pb.Baseline(self.completeDf[item.data()].index, check_finite=False)
            for entry in self.completeDf:
                background = baseline_fitter.pspline_airpls(self.completeDf[entry].values, lam = arPLSLambda)[0]
                self.completeDf[entry] = self.completeDf[entry] - background


    def arPLSSmoothing(self):
        self.arPLS = QDialog()
        #print(os.getcwd())
        loadUi("./arPLSSmoothing.ui", self.arPLS)
        self.arPLS.show()

        self.arPLS.buttonBox.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self.arPLSapply)
        self.arPLS.showPreview.clicked.connect(self.arPLSplot)

    def furrierFilter(self):
        self.furrierFilter = QDialog()
        loadUi("./Furrier filter.ui", self.furrierFilter)
        self.furrierFilter.show()
        self.furrierFilter.buttonBox.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self.furrierSmoothing)
        #self.ffCutoffValue = self.findChild(QDoubleSpinBox, "ffCutoffValue")
        print(f"[furrierFilter] Cutoff value {self.furrierFilter.ffCutoffValue.value()}")

    def weinerFilter(self):
        self.weinerFilter = QDialog()
        loadUi("./Weiner filter.ui", self.weinerFilter)
        self.weinerFilter.show()
        self.weinerFilter.buttonBox.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self.weinerSmoothing)
        print(f"[weinerFilter] Window value: {self.weinerFilter.windowValue.value()}, noise: {self.weinerFilter.noiseValue.value()}")

    def saveCurrentData(self):
        print("saving current data")
        try:
            data = self.completeDf[self.filename].copy()
            # Transforming to raman shift
            # TURNING OFF WAVE TO SHIFT
            # data.index = self.wave_to_shift(wavelength = data.index)
            if self.saveDataFilePath == "":
                fileName, fileType = QFileDialog.getSaveFileName(self, "Save data frame", directory=self.filename, filter="Files (*.csv *.tsv)")
            else:
                fileName, fileType = QFileDialog.getSaveFileName(self, "Save data frame", directory = self.saveDataFilePath+"/"+self.filename, filter="Files (*.csv *.tsv)")

            self.saveDataFilePath = "/".join(fileName.split("/")[0:-1])
            if fileName != "":
                data.to_csv(fileName, sep=",")
            else:
                print("[Action failed] No file is selected. Saving data frame aborted.")
            
        except:
            print("[Action failed] No data is loaded or data is invalid.")

    def savePlotCallback(self):
        if self.savePlotFilePath == "":
            fileName, fileType = QFileDialog.getSaveFileName(self, "Save image", self.filename, "Image Files (*.png *.jpg *.bmp)")
        else:
            fileName, fileType = QFileDialog.getSaveFileName(self, "Save image", directory = self.savePlotFilePath + "/" + self.filename, filter = "Image Files (*.png *.jpg *.bmp)")
        
        self.savePlotFilePath = "/".join(fileName.split("/")[0:-1])
        self.plot.savePlot(fileName)

    def fileSelectionCallback(self, index):
        item = self.model.itemFromIndex(index)
        self.filename = item.text()
        print (f"Selected : {self.filename}")

        #data = self.completeDf[self.filename].copy()
        data = self.completeDf[self.filename]
        #print(f"Data : {data}")
        #print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        # Transforming to raman shift
        # TURNING OFF WAVE TO SHIFT
        index = self.wave_to_shift(wavelength = data.index)
        
        # Checkign if smoothedDF exists. If not, no smoothing is applied.
        try:
            print(f"[fileSelectionCallback] Smoothed data printed")
            self.plot.update_plot(data = data, index=data.index, smoothedData=self.smoothedDf[self.filename].values)
        except:
            self.plot.update_plot(data = data, index=data.index)

    def loadFile(self):
        file, placeholder = QFileDialog.getOpenFileName(self, "Open file", self.lastFolder)
        # self.printFileNames(file)
        print(f"[loadFile] File : {file}")

        if file[-4:] == ".csv":
            self.loadCSVfile(file)
        elif file[-4:] == ".sif":
            self.loadSIFFile(file)
        elif file[-7:] == ".csv.gz":
            self.loadCsvGzipFile(file)
        else:
            print(f"[loadFile] Unknonw file format of input {file}.")
    
    def loadFolder(self):
        #print("Load folder clicked.")
        folder = QFileDialog.getExistingDirectory(self, "Open directory", self.lastFolder )

        self.lastFolder = folder

        files = self.loadFilesIntoMemory(folder)
        self.printFileNames(folder, files)

    def printFileNames(self, folder, files):
        if folder != "":
            #files = os.listdir(folder)
            self.model.clear()
            for file in files:
                if "_xyzPositions" in file:
                    continue
                else:
                    self.model.appendRow(QStandardItem(file))
        else:
            print("[Action failed] No folder selected.")

    def loadCSVfile(self, path):
        self.completeDf = pd.read_csv(path, sep='\t', decimal=',', index_col=0)
        self.completeDf = self.completeDf.transpose()

        newCols = []
        for column in self.completeDf.columns:
            newCols.append(str(column))
        
        self.completeDf.columns = newCols

        # print(f"Data Frame : {self.completeDf}")
        # Clearing old values and adding new ones
        self.model.clear()
        for column in self.completeDf.columns:
            self.model.appendRow(QStandardItem(str(column)))

    def loadSIFFile(self, path):
        print("Attempting to load sif file.")
        # Loading data from sif array. Returns a numpy.ndarray 
        # [[index value 1, column value 1], [index value 2, column value 2], ...]
        # [[  257., 12098.]
        # ...
        # [  768., 15538.]] 
        data = sif2array(target=path, reduce_noise=False, window='narrow')
        # Transforming into pandas data frame and asssigning column 0 as index 
        df1 = pd.DataFrame(data)
        #df1.index = df1[0]
        #del df1[0]

        return df1

    def loadCsvGzipFile(self, path):
        print("Attempting to load gzipped csv file.")

        self.completeDf = pd.read_csv(path, sep=',', decimal='.', compression="gzip", index_col="Wavelengths")
        # Retrieving filename as a name for the spectrum in the UI
        filename = path.split("/")[-1]
        # Filename should also be the column name.
        self.completeDf.columns = [filename]

        self.model.clear()
        self.model.appendRow(QStandardItem(str(filename)))

    def loadFilesIntoMemory(self, folder):
        if folder: 
            allFiles = os.listdir(folder)
            files = []
            firstFile = True
            for file in allFiles:
                if "xyzPositions" in file:
                    continue
                path = folder + "/" + file
                if path[-4:] == ".csv":
                    fileDf = pd.read_csv(path, sep='\t', header=None, decimal=',')
                elif path[-4:] == ".sif":
                    try:
                        fileDf = self.loadSIFFile(path)
                    except:
                        print(f"[loadFilesIntoMemory] File {file} apprears corrupted. Skipping.")
                        continue
                elif path[-7:] == ".csv.gz":
                    #print("Entering .csv.gz section.")
                    try:
                        fileDf = pd.read_csv(path, sep=',', decimal='.', compression="gzip", index_col="Wavelengths")
                        if "Unnamed: 0" in fileDf.columns:
                            # This is a stray index, that we need to remove. 
                            del fileDf["Unnamed: 0"]
                    except OSError as e:
                        print(f"[loadFilesInMemory] Loading files failed. Path: {path}. Error: {e}")
                        quit()
                else:
                    print(f"[loadFilesIntoMemory] Error: Unsupported file format: {file}. Skipping")
                    continue
                files.append(file)
                # Checking if the dataframe contains only one spectrum column or multiple
                if len(fileDf.columns) == 2:
                    if firstFile:
                        self.completeDf = fileDf
                        self.completeDf.reset_index(inplace=True)
                        try:
                            index = self.completeDf[0].copy()
                        except:
                            print("exception!")
                            print(self.completeDf)
                            quit()
                        # Removing index created after reindexing
                        del self.completeDf["index"]
                        # Removing original wavelength column
                        del self.completeDf[0]
                        firstFile = False
                        self.completeDf.index = index
                    else:
                        self.completeDf = pd.concat([self.completeDf, fileDf[1]], axis=1, ignore_index=True)
                elif len(fileDf.columns) == 1:
                    # csv file that has an index row as index and only one spectrum per file.
                    if firstFile:
                        self.completeDf = fileDf
                        firstFile = False
                    else:
                        self.completeDf = pd.concat([self.completeDf, fileDf], axis=1, ignore_index=True)
                else:
                    print("This data frame format is not implemented yet. Bye!")
                    quit()

            self.completeDf.columns = files

            #self.completeDf.index = index

            self.completeDf.to_csv("C:/Users/adato/Programs/complete_dataframe.csv")

            return files

    def exitProgram(self):
        print("User requested program exit.")
        quit()

    def wave_to_shift(self, wavelength : float, excitation_wavelenth = 532.0) -> float:
        """
        Transforming wavelength to a Raman shift. Works also for `list` like objects. 
        `wavelength` is a float or a list of floats of wavelength in **nm**. `excitation_wavelength` 
        is a wavelength of an excitation laser in **nm** (default: 532). Returns a Raman shift
        in reciprocal centimeters or a list thereof.
        """
        shift = (1e7/excitation_wavelenth) - (1e7/wavelength)
        return shift
    
    def frameAnimation(self, frame):
        if frame == "smoothing":
            objectDict = {"Furrier filter": self.furrierFrame}
        else: 
            print("Something.")

        selectedSmoothing = self.smoothingValueCBox.currentText()

        if selectedSmoothing == "No smoothing":
            # Closing all frames.
            for object in objectDict:
                objectDict[object].setMaximumHeight(0)
        else:
            frameObject = objectDict.pop(selectedSmoothing)
            print(f"[FrameAnimation] {selectedSmoothing} selected.")
            # Closing all unselected frames
            if len(objectDict) > 0:
                for object in objectDict:
                    objectDict[object].setMaximumHeight(0)

            duration = 500
            self.animationSequence = QPropertyAnimation(frameObject, b"maximumSize")
            self.animationSequence.setDuration(duration)
            self.animationSequence.setStartValue(frameObject.maximumSize())
            self.animationSequence.setEndValue(QSize(1000,1000))
            
            self.animationSequence.start()
        

        """
        print("Something else selected")
        duration = 250
        self.animationMenu = QPropertyAnimation(self.frameObject, b"maximumSize")
        self.animationMenu.setDuration(duration)
        self.animationMenu.setStartValue(self.frameObject.maximumSize())
        self.animationMenu.setEndValue(QSize(1000,0))
        self.animationMenu.start()
        
        """



if __name__ == "__main__":
    app = QApplication([])

    window = main()
    window.show()

    sys.exit(app.exec())