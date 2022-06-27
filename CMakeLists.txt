cmake_minimum_required(VERSION 3.13.4)

project(MEMOS)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/MEMOS")
set(EXTENSION_CATEGORY "SlicerMorph")
set(EXTENSION_CONTRIBUTORS "Sara Rolfe (SCRI), Murat Maga (SCRI, UW)")
set(EXTENSION_DESCRIPTION "This model loads a PyTorch Deep Learning model and predicts a segmentation with 50 labeled regions on a new scan volume.")
set(EXTENSION_ICONURL "http://www.example.com/Slicer/Extensions/MEMOS.png")
set(EXTENSION_SCREENSHOTURLS "http://www.example.com/Slicer/Extensions/MEMOS/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(MEMOS)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
