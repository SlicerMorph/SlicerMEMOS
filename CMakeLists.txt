cmake_minimum_required(VERSION 3.13.4)

project(MEMOS)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/SlicerMorph/SlicerMEMOS")
set(EXTENSION_CATEGORY "Segmentation")
set(EXTENSION_CONTRIBUTORS "Sara Rolfe (SCRI), Murat Maga (SCRI, UW)")
set(EXTENSION_DESCRIPTION "This extension enables deep learning-powered segmentation of 50 anatomical structures from microCT scans of embryonic mice.")
set(EXTENSION_ICONURL "https://github.com/SlicerMorph/SlicerMEMOS/raw/main/MEMOS/Resources/Icons/MEMOS.png")
set(EXTENSION_SCREENSHOTURLS "https://github.com/SlicerMorph/SlicerMEMOS/raw/main/memos.jpg")
set(EXTENSION_DEPENDS "PyTorch") # Specified as a list or "NA" if no dependencies

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
