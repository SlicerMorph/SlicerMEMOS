import os
import unittest
import logging
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

import logging
import os
import sys
import tempfile
import shutil
import time
from glob import glob
from packaging import version

#
# MEMOS
#
class MEMOS(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
      https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
      """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "MEMOS"  # TODO make this more human readable by adding spaces
        self.parent.categories = ["MEMOS"]
        self.parent.dependencies = []
        self.parent.contributors = [
            "Sara Rolfe (SCRI), Murat Maga (SCRI, UW)"]  # replace with "Firstname Lastname (Organization)"
        self.parent.helpText = """
      This model loads a PyTorch Deep Learning model and does inference on an 3D diceCT scan of a mouse fetus loaded in the scene. For more information, please see online documentation: https://github.com/SlicerMorph/SlicerMEMOS#readme
      """
        self.parent.acknowledgementText = """
      This module was developed by Sara Rolfe and was supported by grants (OD032627 and HD104435) awarded to Murat Maga from National Institutes of Health."
      """  # replace with organization, grant and thanks.
       # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#

def registerSampleData():
    """
    Add data sets to Sample Data module.
    """
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData
    iconsPath = os.path.join(os.path.dirname(__file__), 'Resources/Icons')

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # FastModelAlign1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category='MEMOs',
        sampleName='Sample E15 embryo',
        uris= ["https://raw.githubusercontent.com/SlicerMorph/SampleData/master/IMPC_sample_data.nrrd"],
        checksums= [None],
        loadFiles=[True],
        fileNames=['E15_Sample.nrrd'],
        nodeNames=['E15_Sample'],
        thumbnailFileName=os.path.join(iconsPath, 'MEMOSample.png'),
        loadFileType=['VolumeFile']

    )






class MEMOSWidget(ScriptedLoadableModuleWidget):
    """Uses ScriptedLoadableModuleWidget base class, available at:
      https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
      """
    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        # Set up tabs to split workflow
        tabsWidget = qt.QTabWidget()
        singleTab = qt.QWidget()
        singleTabLayout = qt.QFormLayout(singleTab)
        batchTab = qt.QWidget()
        batchTabLayout = qt.QFormLayout(batchTab)
        tabsWidget.addTab(singleTab, "Single volume")
        tabsWidget.addTab(batchTab, "Batch mode")
        self.layout.addWidget(tabsWidget)

        ################################### Single Tab ###################################
        # Instantiate and connect widgets
        #
        # Segmentation Set-up Area
        #
        singleParametersCollapsibleButton = ctk.ctkCollapsibleButton()
        singleParametersCollapsibleButton.text = "Segmentation Set-up"
        singleTabLayout.addRow(singleParametersCollapsibleButton)

        # Layout within the dummy collapsible button
        singleParametersFormLayout = qt.QFormLayout(singleParametersCollapsibleButton)

        #
        # Select base mesh
        #
        self.volumeSelector = slicer.qMRMLNodeComboBox()
        self.volumeSelector.nodeTypes = ( ("vtkMRMLVolumeNode"), "" )
        self.volumeSelector.selectNodeUponCreation = False
        self.volumeSelector.addEnabled = False
        self.volumeSelector.removeEnabled = False
        self.volumeSelector.noneEnabled = True
        self.volumeSelector.showHidden = False
        self.volumeSelector.setMRMLScene( slicer.mrmlScene )
        singleParametersFormLayout.addRow("Volume: ", self.volumeSelector)

        #
        # Select model file
        #
        self.modelPathSingle = ctk.ctkPathLineEdit()
        self.modelPathSingle.currentPath = self.getMEMOSModelPath()
        self.modelPathSingle.filters = ctk.ctkPathLineEdit.Files
        self.modelPathSingle.nameFilters= ["Model (*.pth)"]
        self.modelPathSingle.setToolTip("Select the segmentation model")
        singleParametersFormLayout.addRow("Segmentation model: ", self.modelPathSingle)

        #
        # Apply Single Button
        #
        self.applySingleButton = qt.QPushButton("Apply")
        self.applySingleButton.toolTip = "Generate MEMOS segmentation for loaded volume"
        self.applySingleButton.enabled = False
        singleParametersFormLayout.addRow(self.applySingleButton)

        #
        # Evaluation Area
        #
        singleEvaluationCollapsibleButton = ctk.ctkCollapsibleButton()
        singleEvaluationCollapsibleButton.text = "Evaluate Segmentation"

        #
        # Select MEMOS segmentation
        #
        self.MEMOSSelector = slicer.qMRMLNodeComboBox()
        self.MEMOSSelector.nodeTypes = ( ("vtkMRMLSegmentationNode"), "" )
        self.MEMOSSelector.selectNodeUponCreation = False
        self.MEMOSSelector.addEnabled = False
        self.MEMOSSelector.removeEnabled = False
        self.MEMOSSelector.noneEnabled = True
        self.MEMOSSelector.showHidden = False
        self.MEMOSSelector.setMRMLScene( slicer.mrmlScene )

        #
        # Select Reference segmentation
        #
        self.referenceSelector = slicer.qMRMLNodeComboBox()
        self.referenceSelector.nodeTypes = ( ("vtkMRMLSegmentationNode"), "" )
        self.referenceSelector.selectNodeUponCreation = False
        self.referenceSelector.addEnabled = False
        self.referenceSelector.removeEnabled = False
        self.referenceSelector.noneEnabled = True
        self.referenceSelector.showHidden = False
        self.referenceSelector.setMRMLScene( slicer.mrmlScene )

        #
        # Apply Evaluation Button
        #
        self.evaluateSegmentationButton = qt.QPushButton("Compare Segmentations")
        self.evaluateSegmentationButton.toolTip = "Compare MEMOS segmentation to a reference"
        self.evaluateSegmentationButton.enabled = False

        # Connections
        self.volumeSelector.connect('currentNodeChanged(vtkMRMLNode*)', self.onSelectSingle)
        self.modelPathSingle.connect('currentPathChanged(const QString &)', self.onSelectSingleModelPath)
        self.applySingleButton.connect('clicked(bool)', self.onApplySingleButton)
        self.MEMOSSelector.connect('currentNodeChanged(vtkMRMLNode*)', self.onSelectSingleEval)
        self.referenceSelector.connect('currentNodeChanged(vtkMRMLNode*)', self.onSelectSingleEval)

        ################################### Batch Tab ###################################
        # Instantiate and connect widgets
        #
        # Segmentation Set-up Area
        #
        parametersCollapsibleButton = ctk.ctkCollapsibleButton()
        parametersCollapsibleButton.text = "Segmentation Set-up"
        batchTabLayout.addRow(parametersCollapsibleButton)

        # Layout within the dummy collapsible button
        parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)

        #
        # Select volume directory
        #
        self.volumePath = ctk.ctkPathLineEdit()
        self.volumePath.filters = ctk.ctkPathLineEdit.Dirs
        self.volumePath.setToolTip("Select the volume directory")
        parametersFormLayout.addRow("Volume directory: ", self.volumePath)

        #
        # Select model file
        #
        self.modelPath = ctk.ctkPathLineEdit()
        self.modelPath.currentPath = self.getMEMOSModelPath()
        self.modelPath.filters = ctk.ctkPathLineEdit.Files
        self.modelPath.nameFilters= ["Model (*.pth)"]
        self.modelPath.setToolTip("Select the segmentation model")
        parametersFormLayout.addRow("Segmentation model: ", self.modelPath)

        #
        # Select volume directory
        #
        self.outputPath = ctk.ctkPathLineEdit()
        self.outputPath.filters = ctk.ctkPathLineEdit.Dirs
        self.outputPath.setToolTip("Select the output directory")
        parametersFormLayout.addRow("Output directory: ", self.outputPath)

        #
        # Apply Button
        #
        self.applyButton = qt.QPushButton("Apply")
        self.applyButton.toolTip = "Generate MEMOS segmentation for each volume in directory."
        self.applyButton.enabled = False
        parametersFormLayout.addRow(self.applyButton)

        # connections
        self.volumePath.connect('currentPathChanged(const QString &)', self.onSelect)
        self.modelPath.connect('currentPathChanged(const QString &)', self.onSelectModelPath)
        self.outputPath.connect('currentPathChanged(const QString &)', self.onSelect)
        self.applyButton.connect('clicked(bool)', self.onApplyButton)

        # Add vertical spacer
        self.layout.addStretch(1)

    def onSelectSingle(self):
        self.applySingleButton.enabled = bool(self.volumeSelector.currentNode() and self.modelPathSingle.currentPath)

    def onSelectSingleModelPath(self):
        self.saveMEMOSModelPath(self.modelPathSingle.currentPath)
        self.modelPath.currentPath = self.getMEMOSModelPath()
        self.applySingleButton.enabled = bool(self.volumeSelector.currentNode() and self.modelPathSingle.currentPath)

    def onSelectSingleEval(self):
        self.evaluateSegmentationButton.enabled = bool(self.MEMOSSelector.currentNode() and self.referenceSelector.currentNode())

    def onSelect(self):
        self.applyButton.enabled = bool(self.modelPath.currentPath and self.volumePath.currentPath and self.outputPath.currentPath)

    def onSelectModelPath(self):
        self.saveMEMOSModelPath(self.modelPathSingle.currentPath)
        self.modelPathSingle.currentPath = self.getMEMOSModelPath()
        self.applyButton.enabled = bool(self.modelPath.currentPath and self.volumePath.currentPath and self.outputPath.currentPath)

    def onApplyButton(self):
        logic = MEMOSLogic()
        logic.setupPythonRequirements()
        self.setColorTable()
        volumeDir = self.volumePath.currentPath
        outputPath = self.outputPath.currentPath
        # get volumes in directory
        images = []
        volumeExtensions = ['nrrd', 'nii.gz']
        for root, dirnames, imageNames in os.walk(volumeDir):
          for imageName in imageNames:
            imName, imExt = imageName.split(os.extsep,1)
            if any( ext in imExt for ext in volumeExtensions):
              images.append(os.path.join(root, imageName))
        for image in images:
          volumeNode = slicer.util.loadVolume(image)
          labelFilePath = self.launchInference(volumeNode)
          # load resulting labelmap
          try:
            labelNode = slicer.util.loadLabelVolume(labelFilePath)
          except:
            print("No segmentation generated for ", volumeNode.GetName())
            return
          dataType = labelNode.GetImageData().GetScalarTypeAsString()
          if dataType != "unsigned char":
            logic.castVolumeToUnsignedChar(labelNode)
          labelNode.SetSpacing(volumeNode.GetSpacing())
          labelNode.SetOrigin(volumeNode.GetOrigin())
          labelNode.GetDisplayNode().SetAndObserveColorNodeID(self.colorNode.GetID())
          # Create segmentation node from label map
          segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
          slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelNode, segmentationNode)
          segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(volumeNode)
          segmentationNode.CreateClosedSurfaceRepresentation()
          outputSegFile = os.path.join(outputPath, volumeNode.GetName() + ".seg.nrrd")
          slicer.util.saveNode(segmentationNode, outputSegFile)
          # Clean up
          slicer.mrmlScene.RemoveNode(labelNode)
          slicer.mrmlScene.RemoveNode(segmentationNode)
          slicer.mrmlScene.RemoveNode(volumeNode)

        tempLabelDir = os.path.dirname(labelFilePath)
        shutil.rmtree(tempLabelDir)

    def onApplySingleButton(self):
      logic = MEMOSLogic()
      logic.setupPythonRequirements()
      self.setColorTable()
      volumeNode = self.volumeSelector.currentNode()
      # Do inference
      labelFilePath = self.launchInference(volumeNode)
      # Load resulting labelmap
      try:
        labelNode = slicer.util.loadLabelVolume(labelFilePath)
      except:
        print("No segmentation generated")
        return
      labelNode.SetSpacing(volumeNode.GetSpacing())
      labelNode.SetOrigin(volumeNode.GetOrigin())
      labelNode.GetDisplayNode().SetAndObserveColorNodeID(self.colorNode.GetID())
      # Create segmentation node from label map
      segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
      slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelNode, segmentationNode)
      segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(volumeNode)
      segmentationNode.CreateClosedSurfaceRepresentation()
      #assign segmentation to volume node
      shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
      volID = shNode.GetItemByDataNode(volumeNode)
      segID = shNode.GetItemByDataNode(segmentationNode)
      shNode.SetItemParent(segID,volID)
      self.MEMOSSelector.setCurrentNode(segmentationNode)
      # Clean up
      slicer.mrmlScene.RemoveNode(labelNode)
      tempLabelDir = os.path.dirname(labelFilePath)
      shutil.rmtree(tempLabelDir)

    def launchInference(self, volumeNode):
      # Make directory to store volume for inference
      logic = MEMOSLogic()
      tempVolumeDir = os.path.join(slicer.app.temporaryPath, 'tempMEMOSVolume')
      if os.path.isdir(tempVolumeDir):
        shutil.rmtree(tempVolumeDir)
      os.mkdir(tempVolumeDir)

      # Temporarily set volume spacing and origin to default values
      originalSpacing = volumeNode.GetSpacing()
      originalOrigin = volumeNode.GetOrigin()
      volumeNode.SetSpacing((1,1,1))
      volumeNode.SetOrigin(0,0,0)
      tempVolumeFile = os.path.join(tempVolumeDir, volumeNode.GetName() +'.nii.gz')
      slicer.util.saveNode(volumeNode, tempVolumeFile)

      # Create directory for output segmentation
      tempOutputPath = os.path.join(slicer.app.temporaryPath,'tempMEMOSOut')
      if os.path.isdir(tempOutputPath):
        shutil.rmtree(tempOutputPath)
      os.mkdir(tempOutputPath)
      outputName = os.path.basename(volumeNode.GetName())
      outputLabelPath = os.path.join(tempOutputPath, outputName + "_seg.nii.gz")

      # run inference
      logic.processInference(tempVolumeFile, self.modelPathSingle.currentPath, outputLabelPath, self.colorNode)

      # Reset the volume spacing and origin to original values
      volumeNode.SetSpacing(originalSpacing)
      volumeNode.SetOrigin(originalOrigin)
      shutil.rmtree(tempVolumeDir)
      return outputLabelPath

#    def onEvaluateSegmentationButton(self):
#      logic = MEMOSLogic()
#      logic.getDiceTable(self.referenceSelector.currentNode(), self.MEMOSSelector.currentNode())

    def setColorTable(self):
      if hasattr(self, 'colorNode'):
        try:
          slicer.util.getNode(self.colorNode)
          return
        except:
          print("Color node is not in the scene, reloading from file")
          self.colorNode = None
      # Load color table
      colorNodePath = self.resourcePath("Support/KOMP2.ctbl")
      try:
        self.colorNode = slicer.util.loadColorTable(colorNodePath)
      except:
        print("Error loading color node from: ", colorNodePath)
        self.colorNode = None

    def downloadMEMOSModel(self):
      # if no valid model path, try to download
      import SampleData
      filename = "best_metric_model_largePatch_noise.pth"
      link = "https://app.box.com/shared/static/4nygg33o70oj5xvnhew11zz5geclus5b.pth"
      progressDialog = slicer.util.createProgressDialog(
                labelText="Downloading the MEMOS segmentation model. This may take a minute.",
                maximum=0,
            )
      slicer.app.processEvents()
      SampleData.SampleDataLogic().downloadFileIntoCache(link, filename)
      progressDialog.close()
      modelPath = os.path.join(slicer.app.cachePath, filename)
      if self.isValidMEMOSModelPath(modelPath):
        self.saveMEMOSModelPath(modelPath)
        return modelPath
      else:
        slicer.util.infoDisplay("Error downloading MEMOS segmentation model. Please download manually following the instructions on the module help page.")
        return ""

    def saveMEMOSModelPath(self, modelPath):
      # don't save if identical to saved
      settings = qt.QSettings()
      if settings.contains('Developer/MEMOSModelPath'):
        modelPathSaved = settings.value('Developer/MEMOSModelPath')
        if modelPathSaved == modelPath:
          return
      if not self.isValidMEMOSModelPath(modelPath):
        return
      settings.setValue('Developer/MEMOSModelPath', modelPath)

    def getMEMOSModelPath(self):
      # If path is defined in settings then use that
      settings = qt.QSettings()
      if settings.contains('Developer/MEMOSModelPath'):
        modelPath = settings.value('Developer/MEMOSModelPath')
        if self.isValidMEMOSModelPath(modelPath):
          return modelPath
      modelPath = self.downloadMEMOSModel()
      return modelPath

    def isValidMEMOSModelPath(self, modelPath):
      if os.path.exists(modelPath):
        return True
      else:
        print("MEMOS model path invalid: No model found")
        return False

#
# MEMOSLogic
#

class MEMOSLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
      computation done by your module.  The interface
      should be such that other python code can import
      this class and make use of the functionality without
      requiring an instance of the Widget.
      Uses ScriptedLoadableModuleLogic base class, available at:
      https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
      """
    def castVolumeToUnsignedChar(self, inputVolumeNode):
      # Set parameters
      parameters = {}
      parameters["InputVolume"] = inputVolumeNode
      parameters["OutputVolume"] = inputVolumeNode
      parameters["Type"]="UnsignedShort"
      # Execute
      caster = slicer.modules.castscalarvolume
      cliNode = slicer.cli.runSync(caster, None, parameters)
      # Process results
      if cliNode.GetStatus() & cliNode.ErrorsMask:
        # error
        errorText = cliNode.GetErrorText()
        slicer.mrmlScene.RemoveNode(cliNode)
        raise ValueError("CLI execution failed: " + errorText)
      # success
      slicer.mrmlScene.RemoveNode(cliNode)
      return

    def processInference(self, volumePath, modelPath, outputLabelPath, colorNode):
      inputVolume = {"image": volumePath}
      # Get path to inference script
      self.moduleDir = os.path.dirname(slicer.util.getModule('MEMOS').path)
      inferenceScriptPyFile = os.path.join(self.moduleDir, "Scripts", "MEMOS_inference.py")
      # Get Python executable path
      pythonSlicerExecutablePath = shutil.which("PythonSlicer")
      if not pythonSlicerExecutablePath:
        raise RuntimeError("Python was not found")
      MEMOS_inferenceCommand = [ pythonSlicerExecutablePath, str(inferenceScriptPyFile),
        "--volume-path", str(inputVolume),
        "--model-path", str(modelPath),
        "--output-path", str(outputLabelPath),
        "--color-node", str(colorNode) ]
      proc = slicer.util.launchConsoleProcess(MEMOS_inferenceCommand)
      slicer.util.logProcessOutput(proc)

    def setupPythonRequirements(self):
      print("Checking python dependencies")
      # Install PyTorch
      try:
        import PyTorchUtils
      except ModuleNotFoundError:
        slicer.util.messageBox("MEMOS requires the PyTorch extension. Please install it from the Extensions Manager.")
      torchLogic = PyTorchUtils.PyTorchUtilsLogic()
      if not torchLogic.torchInstalled():
        logging.debug('MEMOS requires the PyTorch Python package. Installing... (it may take several minutes)')
        torch = torchLogic.installTorch(askConfirmation=True)
        if torch is None:
          slicer.util.messageBox('PyTorch extension needs to be installed manually to use this module.')

      # Install additional python packages
      try:
        import pillow
      except:
        logging.debug('Pillow Python package is required. Installing...')
        slicer.util.pip_install('pillow')
      try:
        import nibabel
      except:
        logging.debug('Nibabel Python package is required. Installing...')
        slicer.util.pip_install('nibabel')
      try:
        import einops
      except:
        logging.debug('Einops Python package is required. Installing...')
        slicer.util.pip_install('einops')

      # Install MONAI and restart if the version was updated.
      monaiVersion = "0.9.0"
      try:
        import monai
        if version.parse(monai.__version__) != version.parse(monaiVersion):
          logging.debug(f'MEMOS requires MONAI version {monaiVersion}. Installing... (it may take several minutes)')
          slicer.util.pip_uninstall('monai')
          slicer.util.pip_install('monai[pynrrd,fire]=='+ monaiVersion)
          if slicer.util.confirmOkCancelDisplay(f'MONAI version was updated {monaiVersion}.\n Click OK restart Slicer.'):
            slicer.util.restart()
      except:
        logging.debug('MEMOS requires installation of the MONAI Python package. Installing... (it may take several minutes)')
        slicer.util.pip_install('monai[pynrrd,fire]=='+ monaiVersion)
      try:
        import fire
      except:
        slicer.util.pip_install('fire')

#    def getDiceTable(self, refSeg, predictedSeg):
#      pnode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentComparisonNode")
#      pnode.SetAndObserveReferenceSegmentationNode(refSeg)
#      pnode.SetAndObserveCompareSegmentationNode(predictedSeg)
#      dtab = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode")
#      fullTable = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode")
#      fullTable.SetName(refSeg.GetName())
#      fullTable.SetUseFirstColumnAsRowHeader(False)
#      header = fullTable.AddColumn()
#      header.SetName("Metric name")
#      header.InsertNextValue("Reference segmentation")
#      header.InsertNextValue("Reference segment")
#      header.InsertNextValue("Compare segmentation")
#      header.InsertNextValue("Compare segment")
#      header.InsertNextValue("Dice coefficient")
#      header.InsertNextValue("True positives (%)")
#      header.InsertNextValue("True negatives (%)")
#      header.InsertNextValue("False positives (%)")
#      header.InsertNextValue("False negatives (%)")
#      header.InsertNextValue("Reference center")
#      header.InsertNextValue("Compare center")
#      header.InsertNextValue("Reference volume (cc)")
#      header.InsertNextValue("Compare volume (cc)")
#      fullTable.Modified()
#      for index in range(0,refSeg.GetSegmentation().GetNumberOfSegments()):
#        pnode.SetAndObserveDiceTableNode(dtab)
#        pnode.SetReferenceSegmentID(refSeg.GetSegmentation().GetNthSegmentID(index))
#        pnode.SetCompareSegmentID(predictedSeg.GetSegmentation().GetNthSegmentID(index))
#        success = slicer.modules.segmentcomparison.logic().ComputeDiceStatistics(pnode)
#        col = dtab.GetTable().GetColumn(1)
#        col.SetName(predictedSeg.GetSegmentation().GetNthSegmentID(index+1))
#        fullTable.AddColumn(col)
#      flipper = vtk.vtkTransposeTable()
#      flipper.SetInputData(fullTable.GetTable())
#      flipper.Update()
#      fullTable.SetAndObserveTable(flipper.GetOutput())
#      fullTable.RemoveColumn(0)
#      fullTable.RemoveColumn(0)
#      fullTable.RemoveColumn(0)
#      fullTable.RemoveColumn(0)
#      fullTable.SetUseFirstColumnAsRowHeader(True)
#      fullTable.SetUseColumnNameAsColumnHeader(True)
#      slicer.mrmlScene.RemoveNode(dtab)

class MEMOSTest(ScriptedLoadableModuleTest):
    """
      This is the test case for your scripted module.
      Uses ScriptedLoadableModuleTest base class, available at:
      https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
      """

    def runTest(self):
        """Run as few or as many tests as needed here.
          """
        self.setUp()
        self.test_MEMOS1()

    def test_MEMOS1(self):
        """ Ideally you should have several levels of tests.  At the lowest level
          tests should exercise the functionality of the logic with different inputs
          (both valid and invalid).  At higher levels your tests should emulate the
          way the user would interact with your code and confirm that it still works
          the way you intended.
          One of the most important features of the tests is that it should alert other
          developers when their changes will have an impact on the behavior of your
          module.  For example, if a developer removes a feature that you depend on,
          your test should break so they know that the feature is needed.
          """

        self.delayDisplay("Starting the test")
        #
        # first, get some data
        #
        import urllib
        downloads = (
            ('http://slicer.kitware.com/midas3/download?items=5767', 'FA.nrrd', slicer.util.loadVolume),
        )
        for url, name, loader in downloads:
            filePath = slicer.app.temporaryPath + '/' + name
            if not os.path.exists(filePath) or os.stat(filePath).st_size == 0:
                logging.info(f'Requesting download {name} from {url}...\n')
                urllib.urlretrieve(url, filePath)
            if loader:
                logging.info(f'Loading {name}...')
                loader(filePath)
        self.delayDisplay('Finished with download and loading')

        volumeNode = slicer.util.getNode(pattern="FA")
        logic = MEMOSLogic()
        self.assertIsNotNone(logic.hasImageData(volumeNode))
        self.delayDisplay('Test passed!')
