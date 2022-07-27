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
from glob import glob

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
        self.parent.categories = ["SlicerMorph.SlicerMorph Utilities"]
        self.parent.dependencies = []
        self.parent.contributors = [
            "Sara Rolfe (UW), Murat Maga (UW)"]  # replace with "Firstname Lastname (Organization)"
        self.parent.helpText = """
      This model loads a PyTorch Deep Learning model and does inference on an image loaded in the scene.
      """
        self.parent.acknowledgementText = """
      This module was developed by Sara Rolfe for SlicerMorph. SlicerMorph was originally supported by an NSF/DBI grant, "An Integrated Platform for Retrieval, Visualization and Analysis of 3D Morphology From Digital Biological Collections" 
      awarded to Murat Maga (1759883), Adam Summers (1759637), and Douglas Boyer (1759839). 
      https://nsf.gov/awardsearch/showAward?AWD_ID=1759883&HistoricalAwards=false
      """  # replace with organization, grant and thanks.

class MEMOSWidget(ScriptedLoadableModuleWidget):
    """Uses ScriptedLoadableModuleWidget base class, available at:
      https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
      """
    def installLibs(self):
      needRestart = False
      # MONAI
      monaiVersion = "0.8.0"
      try:
        import monai
        if version.parse(monai.__version__) != version.parse(monaiVersion):
          slicer.util.pip_uninstall('monai')
          needRestart=1
          if not slicer.util.confirmOkCancelDisplay(f"MEMO requires installation of MONAI (version {monaiVersion}).\nClick OK to install this version and restart Slicer."):
            self.showBrowserOnEnter = False
            return
          slicer.util.pip_install('monai=='+ monaiVersion)
      except:
        slicer.util.pip_install('monai=='+ monaiVersion)
      try:
        import pillow
      except:
        slicer.util.pip_install('pillow')
      try:
        import nibabel
      except:
        slicer.util.pip_install('nibabel')
      try:
        import einops
      except:
        slicer.util.pip_install('einops')
      try:
        import torch
      except:
        if slicer.app.os == 'macosx':
          slicer.util.pip_install('torch torchvision torchaudio')
        elif slicer.app.os == 'linux':
          slicer.util.pip_install('torch==1.10.2+cpu torchvision==0.11.3+cpu torchaudio==0.10.2+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html')
        elif slicer.app.os == 'win':
          slicer.util.pip_install('torch torchvision torchaudio')
        else:
          print("OS not recognized: Dependencies will not be installed")
      if needRestart:
        slicer.util.restart()
    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.installLibs()
        # Instantiate and connect widgets
        #
        # Parameters Area
        #
        parametersCollapsibleButton = ctk.ctkCollapsibleButton()
        parametersCollapsibleButton.text = "Parameters"
        self.layout.addWidget(parametersCollapsibleButton)

        # Layout within the dummy collapsible button
        parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)

        #
        # Select base mesh
        #
        # self.volumeSelector = slicer.qMRMLNodeComboBox()
        # self.volumeSelector.nodeTypes = ( ("vtkMRMLModelNode"), "" )
        # self.volumeSelector.selectNodeUponCreation = False
        # self.volumeSelector.addEnabled = False
        # self.volumeSelector.removeEnabled = False
        # self.volumeSelector.noneEnabled = True
        # self.volumeSelector.showHidden = False
        # self.volumeSelector.setMRMLScene( slicer.mrmlScene )
        # parametersFormLayout.addRow("Base mesh: ", self.volumeSelector)

        #
        # Select volume directory
        #
        self.volumePath = ctk.ctkPathLineEdit()
        self.volumePath.filters = ctk.ctkPathLineEdit.Dirs
        # self.volumePath.nameFilters= ["Volume (*.nrrd)"]
        self.volumePath.setToolTip("Select the volume directory")
        parametersFormLayout.addRow("Volume directory: ", self.volumePath)

        #
        # Select model file
        #
        self.modelPath = ctk.ctkPathLineEdit()
        self.modelPath.filters = ctk.ctkPathLineEdit.Dirs
        # self.modelPath.nameFilters= ["Model (*.pth)"]
        self.modelPath.setToolTip("Select the segmentation model directory")
        parametersFormLayout.addRow("Model directory: ", self.modelPath)

        #
        # Select volume directory
        #
        self.outputPath = ctk.ctkPathLineEdit()
        self.outputPath.filters = ctk.ctkPathLineEdit.Dirs
        # self.outputPath.nameFilters= ["Volume (*.nrrd)"]
        self.outputPath.setToolTip("Select the output directory")
        parametersFormLayout.addRow("Output directory: ", self.outputPath)

        #
        # Apply Button
        #
        self.applyButton = qt.QPushButton("Apply")
        self.applyButton.toolTip = "Generate MEMOSs."
        self.applyButton.enabled = False
        parametersFormLayout.addRow(self.applyButton)

        # connections
        self.volumePath.connect('currentPathChanged(const QString &)', self.onSelect)
        self.modelPath.connect('currentPathChanged(const QString &)', self.onSelect)
        self.outputPath.connect('currentPathChanged(const QString &)', self.onSelect)
        self.applyButton.connect('clicked(bool)', self.onApplyButton)

        # Add vertical spacer
        self.layout.addStretch(1)

    def onSelect(self):
        self.applyButton.enabled = bool(self.modelPath.currentPath and self.volumePath.currentPath
                                        and self.outputPath.currentPath)

    def onApplyButton(self):
        logic = MEMOSLogic()
        logic.run(self.volumePath.currentPath, self.modelPath.currentPath, self.outputPath.currentPath)


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

    def run(self, volumePath, modelPath, outputPath):
        # import MONAI and dependencies 
        import nibabel as nib
        import numpy as np
        import torch
        import einops

        from monai.config import print_config
        from monai.data import Dataset, DataLoader, create_test_image_3d, decollate_batch
        from monai.inferers import sliding_window_inference
        from monai.networks.nets import UNETR
        from monai.data.nifti_writer import write_nifti
        
        from monai.transforms import (
          Activationsd,
          AsDiscreted,
          AddChanneld,
          Compose,
          EnsureChannelFirstd,
          Invertd,
          LoadImaged,
          Orientationd,
          SaveImaged,
          Spacingd,
          CropForegroundd,
          EnsureTyped,
          ScaleIntensityRanged,
          ToTensord
)
        # check configuration
        print_config()

        # define pre transforms
        pre_transforms = Compose([
            LoadImaged(keys=["image"]),
            Spacingd(keys=["image"], pixdim=(
                1, 1, 1), mode="bilinear"),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys="image", axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250,
                b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=["image"], source_key="image"),
            AddChanneld(keys=["image"]),
            ToTensord(keys=["image"]),
        ])

        # define dataset and dataloader
        images = sorted(glob(os.path.join(volumePath, "*.nii.gz")))
        files = [{"image": image} for image in images]
        print("files: ", files)
        # dataset = Dataset(data=files, transform=pre_transforms)
        # dataloader = DataLoader(dataset, batch_size=1, num_workers=0)

        # set up model
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        torch.set_num_threads(24)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device: ", device)

        net = UNETR(
            in_channels=1,
            out_channels=51,
            img_size=(96, 96, 96),
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            pos_embed="perceptron",
            norm_name="instance",
            res_block=True,
            dropout_rate=0.0,
        ).to(device)

        if device.type == "cpu":
            net.load_state_dict(torch.load(os.path.join(modelPath, "best_metric_model.pth"), map_location='cpu'))
        else:
            net.load_state_dict(torch.load(os.path.join(modelPath, "best_metric_model.pth")))
        net.eval()

        with torch.no_grad():
          for filepath in files:
            image = pre_transforms(filepath)['image'].to(device)
            output_raw = sliding_window_inference(
              image, (96, 96, 96), 4, net, overlap=0.8)
            output_formatted = torch.argmax(output_raw, dim=1).detach().cpu()[0, :, :, :]
            outputFile = os.path.basename(filepath.get("image").split('.')[0]) + "_seg.nii.gz"
            print("Writing: ", outputFile)
            write_nifti(
              data=output_formatted,
              file_name=os.path.join(outputPath, outputFile)
              )


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
