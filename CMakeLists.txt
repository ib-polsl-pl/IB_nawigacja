cmake_minimum_required(VERSION 3.13.4)

project(IB_nawigacja)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/IB_nawigacja")
set(EXTENSION_CATEGORY "IGT")
set(EXTENSION_CONTRIBUTORS "Bartlomiej Pycinski (Silesian Univ Tech, Poland)")
set(EXTENSION_DESCRIPTION "A set of IGT modules")
set(EXTENSION_ICONURL "http://www.example.com/Slicer/Extensions/IB_nawigacja.png")
set(EXTENSION_SCREENSHOTURLS "http://www.example.com/Slicer/Extensions/IB_nawigacja/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(ModelOffset)
add_subdirectory(TableRegistration)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
