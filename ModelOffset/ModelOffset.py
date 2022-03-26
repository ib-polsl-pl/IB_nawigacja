import json
import os
import unittest
import logging
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
import numpy as np

#
# ModelOffset
#

class ModelOffset(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "Apply Offset to Model"  # TODO: make this more human readable by adding spaces
    self.parent.categories = ["IGT"]  
    self.parent.dependencies = []  # TODO: add here list of module names that this module requires
    self.parent.contributors = ["Bartłomiej Pyciński (Silesian Univ Tech, Poland)"]  
    self.parent.helpText = """
A module to apply X or Y offset to a model.
See more information in <a href="https://github.com/ib-polsl-pl/ModelOffset.git">module documentation</a>.
"""
    # TODO: replace with organization, grant and thanks
    self.parent.acknowledgementText = """TBD.
"""

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

  # ModelOffset1
  SampleData.SampleDataLogic.registerCustomSampleDataSource(
    # Category and sample name displayed in Sample Data module
    category='ModelOffset',
    sampleName='ModelOffset1',
    # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
    # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
    thumbnailFileName=os.path.join(iconsPath, 'ModelOffset1.png'),
    # Download URL and target file name
    uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
    fileNames='ModelOffset1.nrrd',
    # Checksum to ensure file integrity. Can be computed by this command:
    #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
    checksums = 'SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95',
    # This node name will be used when the data set is loaded
    nodeNames='ModelOffset1'
  )

  # ModelOffset2
  SampleData.SampleDataLogic.registerCustomSampleDataSource(
    # Category and sample name displayed in Sample Data module
    category='ModelOffset',
    sampleName='ModelOffset2',
    thumbnailFileName=os.path.join(iconsPath, 'ModelOffset2.png'),
    # Download URL and target file name
    uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
    fileNames='ModelOffset2.nrrd',
    checksums = 'SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97',
    # This node name will be used when the data set is loaded
    nodeNames='ModelOffset2'
  )

#
# ModelOffsetWidget
#

class ModelOffsetWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent=None):
    """
    Called when the user opens the module the first time and the widget is initialized.
    """
    ScriptedLoadableModuleWidget.__init__(self, parent)
    VTKObservationMixin.__init__(self)  # needed for parameter node observation
    self.logic = None
    self._parameterNode = None
    self._updatingGUIFromParameterNode = False
    self.shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
    self.str_separator = '^'

  def setup(self):
    """
    Called when the user opens the module the first time and the widget is initialized.
    """
    ScriptedLoadableModuleWidget.setup(self)

    # Load widget from .ui file (created by Qt Designer).
    # Additional widgets can be instantiated manually and added to self.layout.
    uiWidget = slicer.util.loadUI(self.resourcePath('UI/ModelOffset.ui'))
    self.layout.addWidget(uiWidget)
    self.ui = slicer.util.childWidgetVariables(uiWidget)

    # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
    # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
    # "setMRMLScene(vtkMRMLScene*)" slot.
    uiWidget.setMRMLScene(slicer.mrmlScene)

    # Create logic class. Logic implements all computations that should be possible to run
    # in batch mode, without a graphical user interface.
    self.logic = ModelOffsetLogic()

    # Connections

    # These connections ensure that we update parameter node when scene is closed
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

    # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
    # (in the selected parameter node).
    # BPWARNING add line here if a new widget is added
    self.ui.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
    self.ui.toolTipTransformSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
    self.ui.controlPointsSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
    self.ui.appliedModelsTreeView.connect("currentItemChanged(vtkIdType)", self.updateParameterNodeFromGUI)
    self.ui.outputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
    self.ui.imageThresholdSliderWidget.connect("valueChanged(double)", self.updateParameterNodeFromGUI)
    self.ui.invertOutputCheckBox.connect("toggled(bool)", self.updateParameterNodeFromGUI)
    self.ui.invertedOutputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)

    # Buttons
    self.ui.applyButton.connect('clicked(bool)', self.onApplyButton)

    # Make sure parameter node is initialized (needed for module reload)
    self.initializeParameterNode()

  def cleanup(self):
    """
    Called when the application closes and the module widget is destroyed.
    """
    self.removeObservers()

  def enter(self):
    """
    Called each time the user opens this module.
    """
    # Make sure parameter node exists and observed
    self.initializeParameterNode()

  def exit(self):
    """
    Called each time the user opens a different module.
    """
    # Do not react to parameter node changes (GUI wlil be updated when the user enters into the module)
    self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

  def onSceneStartClose(self, caller, event):
    """
    Called just before the scene is closed.
    """
    # Parameter node will be reset, do not use it anymore
    self.setParameterNode(None)

  def onSceneEndClose(self, caller, event):
    """
    Called just after the scene is closed.
    """
    # If this module is shown while the scene is closed then recreate a new parameter node immediately
    if self.parent.isEntered:
      self.initializeParameterNode()

  def initializeParameterNode(self):
    """
    Ensure parameter node exists and observed.
    """
    # Parameter node stores all user choices in parameter values, node selections, etc.
    # so that when the scene is saved and reloaded, these settings are restored.

    self.setParameterNode(self.logic.getParameterNode())

    # Select default input nodes if nothing is selected yet to save a few clicks for the user
    if not self._parameterNode.GetNodeReference("InputVolume"):
      firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
      if firstVolumeNode:
        self._parameterNode.SetNodeReferenceID("InputVolume", firstVolumeNode.GetID())
    # BPWARNING
    # we can duplicate the lines above only if we want to pre-fill other combo-boxes

    self.ui.appliedModelsTreeView.nodeTypes = ["vtkMRMLModelNode"]
    # Other possibilities:
    # self.ui.appliedModelsTreeView.sortFilterProxyModel().setNodeTypes(["vtkMRMLScalarVolumeNode"])
    # self.ui.appliedModelsTreeView.sortFilterProxyModel().setNodeTypes(["vtkMRMLModelNode"])


  def setParameterNode(self, inputParameterNode):
    """
    Set and observe parameter node.
    Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
    """

    if inputParameterNode:
      self.logic.setDefaultParameters(inputParameterNode)

    # Unobserve previously selected parameter node and add an observer to the newly selected.
    # Changes of parameter node are observed so that whenever parameters are changed by a script or any other module
    # those are reflected immediately in the GUI.
    if self._parameterNode is not None:
      self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
    self._parameterNode = inputParameterNode
    if self._parameterNode is not None:
      self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

    # Initial GUI update
    self.updateGUIFromParameterNode()

  def updateGUIFromParameterNode(self, caller=None, event=None):
    """
    This method is called whenever parameter node is changed.
    The module GUI is updated to show the current state of the parameter node.
    """

    if self._parameterNode is None or self._updatingGUIFromParameterNode:
      return

    # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
    self._updatingGUIFromParameterNode = True

    # Update node selectors and sliders
    # BPWARNING add line here if a new widget is added
    self.ui.inputSelector.setCurrentNode(self._parameterNode.GetNodeReference("InputVolume"))
    self.ui.toolTipTransformSelector.setCurrentNode(self._parameterNode.GetNodeReference("toolTipTransform"))
    self.ui.controlPointsSelector.setCurrentNode(self._parameterNode.GetNodeReference("controlPoints"))
    self.ui.outputSelector.setCurrentNode(self._parameterNode.GetNodeReference("OutputVolume"))
    self.ui.invertedOutputSelector.setCurrentNode(self._parameterNode.GetNodeReference("OutputVolumeInverse"))
    self.ui.imageThresholdSliderWidget.value = float(self._parameterNode.GetParameter("Threshold"))
    self.ui.invertOutputCheckBox.checked = (self._parameterNode.GetParameter("Invert") == "true")
    # there is a lot of mess with a proper setting of selected models:
    currentItems = vtk.vtkIdList()
    currentItemsIds = self._parameterNode.GetNodeReferenceID("SelectedModels")
    if currentItemsIds is not None:
      currentItemsIds = currentItemsIds.split(self.str_separator)
      for it in currentItemsIds:
          it_id = slicer.util.getNode(it)
          x = self.shNode.GetItemByDataNode(it_id)
          currentItems.InsertNextId(x)
      self.ui.appliedModelsTreeView.setCurrentItems(currentItems)

    # Update buttons states and tooltips
    if self._parameterNode.GetNodeReference("InputVolume") and self._parameterNode.GetNodeReference("toolTipTransform")\
            and self._parameterNode.GetNodeReference("controlPoints") and self._parameterNode.GetNodeReferenceID("SelectedModels"):
      self.ui.applyButton.toolTip = "Translate models"
      self.ui.applyButton.enabled = True
    else:
      self.ui.applyButton.toolTip = "Select all input data"
      self.ui.applyButton.enabled = False

    # All the GUI updates are done
    self._updatingGUIFromParameterNode = False

  def updateParameterNodeFromGUI(self, caller=None, event=None):
    """
    This method is called when the user makes any change in the GUI.
    The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
    """

    if self._parameterNode is None or self._updatingGUIFromParameterNode:
      return

    wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch

    # BPWARNING add line here if a new widget is added
    self._parameterNode.SetNodeReferenceID("InputVolume", self.ui.inputSelector.currentNodeID)
    self._parameterNode.SetNodeReferenceID("toolTipTransform", self.ui.toolTipTransformSelector.currentNodeID)
    self._parameterNode.SetNodeReferenceID("controlPoints", self.ui.controlPointsSelector.currentNodeID)
    self._parameterNode.SetNodeReferenceID("OutputVolume", self.ui.outputSelector.currentNodeID)
    self._parameterNode.SetParameter("Threshold", str(self.ui.imageThresholdSliderWidget.value))
    self._parameterNode.SetParameter("Invert", "true" if self.ui.invertOutputCheckBox.checked else "false")
    self._parameterNode.SetNodeReferenceID("OutputVolumeInverse", self.ui.invertedOutputSelector.currentNodeID)
    # there is a lot of mess with a proper setting of selected models:
    currentItems = vtk.vtkIdList()
    self.ui.appliedModelsTreeView.currentItems(currentItems)
    # we can probably also use `self.ui.appliedModelsTreeView.selectedIndexes()`, but it looks more complicated
    currentItemsIDs = []
    for i in range(currentItems.GetNumberOfIds()):
      vtkId = currentItems.GetId(i)
      try:
        currentItemsIDs.append(self.shNode.GetItemDataNode(vtkId).GetID())
      except AttributeError as e:  # it happens when the last item is unchecked
        print(e)
    self._parameterNode.SetNodeReferenceID("SelectedModels", self.str_separator.join(currentItemsIDs))

    self._parameterNode.EndModify(wasModified)

  def onApplyButton(self):
    """
    Run processing when user clicks "Apply" button.
    """
    try:
      selectedItems = vtk.vtkIdList()
      self.ui.appliedModelsTreeView.currentItems(selectedItems)
      # Compute output
      self.logic.process(self.ui.inputSelector.currentNode(), self.ui.toolTipTransformSelector.currentNode(),
        self.ui.controlPointsSelector.currentNode(), selectedItems, self.ui.offsetXCheckBox.checked, self.ui.offsetYCheckBox.checked,
        self.ui.offsetZCheckBox.checked, self.ui.offsetAllModelsCheckBox.checked)

      # Compute inverted output (if needed)
      if self.ui.invertedOutputSelector.currentNode():
        pass
        # If additional output volume is selected then result with inverted threshold is written there
        #self.logic.process(self.ui.inputSelector.currentNode(), self.ui.invertedOutputSelector.currentNode(),
        #  self.ui.imageThresholdSliderWidget.value, not self.ui.invertOutputCheckBox.checked, showResult=False)

    except Exception as e:
      slicer.util.errorDisplay("Failed to compute results: "+str(e))
      import traceback
      traceback.print_exc()


#
# ModelOffsetLogic
#

class ModelOffsetLogic(ScriptedLoadableModuleLogic):
  """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self):
    """
    Called when the logic class is instantiated. Can be used for initializing member variables.
    """
    ScriptedLoadableModuleLogic.__init__(self)
    self.shNode = slicer.mrmlScene.GetSubjectHierarchyNode()

  def setDefaultParameters(self, parameterNode):
    """
    Initialize parameter node with default settings.
    """
    if not parameterNode.GetParameter("Threshold"):
      parameterNode.SetParameter("Threshold", "100.0")
    if not parameterNode.GetParameter("Invert"):
      parameterNode.SetParameter("Invert", "false")

  def process(self, inputVolume, toolTipTransform, controlPoints, selectedModels, alongX, alongY, alongZ, allModels):
    """
    Run the processing algorithm.
    Can be used without GUI widget.
    :param inputVolume: volume to be thresholded
    :param toolTipTransform: location of a tip of a tool
    :param controlPoints: sequence of the points
    :param vtk.vtkIdList selectedModels : list of models which should be transformed
    :param outputVolume: thresholding result
    :param imageThreshold: values above/below this threshold will be set to 0
    :param invert: if True then values above the threshold will be set to 0, otherwise values below are set to 0
    :param showResult: show output volume in slice viewers
    """


    if not inputVolume or not toolTipTransform or not controlPoints or not selectedModels:
      raise ValueError("Input data are invalid")

    selectedItemsIDs = []
    for i in range(selectedModels.GetNumberOfIds()):
      vtkId = selectedModels.GetId(i)
      selectedItemsIDs.append(self.shNode.GetItemDataNode(vtkId).GetID())

    #toolTipTransform = slicer.util.getNode('StylusTipToStylus')
    tip_mat = vtk.vtkMatrix4x4()
    toolTipTransform.GetMatrixTransformToWorld(tip_mat)
    tip_location = np.array([tip_mat.GetElement(0, 3), tip_mat.GetElement(1, 3), tip_mat.GetElement(2, 3)])
    # tip_transf = vtk.vtkGeneralTransform()
    # toolTipTransform.GetTransformToWorld(tip_transf)

    points = []
    p = [np.nan, np.nan, np.nan]
    for i in range(controlPoints.GetNumberOfControlPoints()):
      controlPoints.GetNthControlPointPosition(i, p)
      points.append(p[:])  # deep-copy the `p`!

    target_point = self.get_nearest_point(tip_location, points)

    offset = tip_location - target_point

    offset *= [alongX, alongY, alongZ]   # apply a mask of bools

    finalTransform = vtk.vtkTransform()
    finalTransform.Translate(offset)
    new_transformNode = slicer.vtkMRMLTransformNode()
    new_transformNode.SetName("Offset_from_a_plugin")
    new_transformNode.SetAndObserveTransformToParent(finalTransform)
    slicer.mrmlScene.AddNode(new_transformNode)

    # BPWARN - all the selected models should have the same parent transform
    old_parent_transform_node = slicer.util.getNode(selectedItemsIDs[0]).GetParentTransformNode()
    if old_parent_transform_node:
      if old_parent_transform_node.GetName() == "Offset_from_a_plugin":
        slicer.mrmlScene.RemoveNode(old_parent_transform_node)
      else:
        new_transformNode.SetAndObserveTransformNodeID(old_parent_transform_node.GetID())
        #model_node.SetAndObserveTransformNodeID(new_transformNode)

        # transformNode2.SetAndObserveTransformNodeID(transformNode1.GetID())
        # transformableNode.SetAndObserveTransformNodeID(transformNode2.GetID())


    for itemId in selectedItemsIDs:
      model_node = slicer.util.getNode(itemId)
      model_node.SetAndObserveTransformNodeID(new_transformNode.GetID())



    print(offset)




    import time
    startTime = time.time()
    logging.info('Processing started')

    # Compute the thresholded output volume using the "Threshold Scalar Volume" CLI module
    cliParams = {
      'InputVolume': inputVolume.GetID(),
      'OutputVolume': outputVolume.GetID(),
      'ThresholdValue' : imageThreshold,
      'ThresholdType' : 'Above' if invert else 'Below'
      }
    cliNode = slicer.cli.run(slicer.modules.thresholdscalarvolume, None, cliParams, wait_for_completion=True, update_display=showResult)
    # We don't need the CLI module node anymore, remove it to not clutter the scene with it
    slicer.mrmlScene.RemoveNode(cliNode)

    stopTime = time.time()
    logging.info('Processing completed in {0:.2f} seconds'.format(stopTime-startTime))

  def load_ureter_coords(self, input_file: str):
    if not input_file.endswith('.json'):
        raise RuntimeError("only Slicer-specific jsons supported")
    with open(input_file) as f:
      point_struct = json.loads(f.read())
      controlPoints = point_struct['markups'][0]['controlPoints']
      points = [p['position'] for p in controlPoints]
      # points = np.array(points)
      return points

  def get_nearest_point(self, current_location, control_points):
    """
    :param current_location: [x y z] coordinates
    :param control_points: [[x1 y1 z1] ... [xn yn zn]]
    """
    import numpy as np
    current_location = np.array(current_location)
    z = current_location[2]
    control_points = np.array(control_points)
    assert (control_points.ndim == 2 and control_points.shape[1] == 3), "Wrong dimension of control points"
    # sort the files along third dimension
    control_points = control_points[np.lexsort([control_points[:, 2]])]
    if len(np.unique(control_points[:, 2])) < len(control_points):
      raise ValueError("Control points are ill-formed. Only one point per Z-slice is allowed.")

    min_z = control_points[0, 2]  # first row
    max_z = control_points[-1, 2]  # last row

    if z <= min_z:
      return control_points[0]

    if z >= max_z:
      return control_points[-1]

    if np.where(control_points[:,2] == z)[0]:
      return control_points[ np.where(control_points[:,2] == z)[0][0] ]

    ind = np.searchsorted(control_points[:,2], z)

    prev_row = control_points[ind-1]
    next_row = control_points[ind]

    x = np.interp(z, [prev_row[2], next_row[2]], [prev_row[0], next_row[0]])
    y = np.interp(z, [prev_row[2], next_row[2]], [prev_row[1], next_row[1]])

    return np.array([x, y, z])

#
# ModelOffsetTest
#

class ModelOffsetTest(ScriptedLoadableModuleTest):
  """
  This is the test case for your scripted module.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setUp(self):
    """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
    slicer.mrmlScene.Clear()

  def runTest(self):
    """Run as few or as many tests as needed here.
    """
    self.setUp()
    self.test_ModelOffset1()

  def test_ModelOffset1(self):
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

    # Get/create input data

    import SampleData
    registerSampleData()
    inputVolume = SampleData.downloadSample('ModelOffset1')
    self.delayDisplay('Loaded test data set')

    inputScalarRange = inputVolume.GetImageData().GetScalarRange()
    self.assertEqual(inputScalarRange[0], 0)
    self.assertEqual(inputScalarRange[1], 695)

    outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
    threshold = 100

    # Test the module logic

    logic = ModelOffsetLogic()

    # Test algorithm with non-inverted threshold
    logic.process(inputVolume, outputVolume, threshold, True)
    outputScalarRange = outputVolume.GetImageData().GetScalarRange()
    self.assertEqual(outputScalarRange[0], inputScalarRange[0])
    self.assertEqual(outputScalarRange[1], threshold)

    # Test algorithm with inverted threshold
    logic.process(inputVolume, outputVolume, threshold, False)
    outputScalarRange = outputVolume.GetImageData().GetScalarRange()
    self.assertEqual(outputScalarRange[0], inputScalarRange[0])
    self.assertEqual(outputScalarRange[1], inputScalarRange[1])

    self.delayDisplay('Test passed')
