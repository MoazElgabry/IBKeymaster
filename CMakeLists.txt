cmake_minimum_required(VERSION 3.20)

project(IBKeyer LANGUAGES CXX)

set(PLUGIN_BASENAME "IBKeyer")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(APPLE AND NOT DEFINED CMAKE_OSX_ARCHITECTURES)
  # The old Makefile built a universal macOS binary for both Intel and Apple Silicon.
  # Keeping that default here avoids an easy regression where the new CMake path silently
  # produces a host-architecture-only plugin depending on which Mac happened to build it.
  set(CMAKE_OSX_ARCHITECTURES "arm64;x86_64" CACHE STRING "macOS architectures to build")
endif()

set(OFX_SUPPORT_SOURCES
  Support/Library/ofxsCore.cpp
  Support/Library/ofxsImageEffect.cpp
  Support/Library/ofxsInteract.cpp
  Support/Library/ofxsLog.cpp
  Support/Library/ofxsMultiThread.cpp
  Support/Library/ofxsParams.cpp
  Support/Library/ofxsProperty.cpp
  Support/Library/ofxsPropertyValidation.cpp
)

set(PLUGIN_SOURCES
  IBKeymaster/IBKeyer.cpp
  IBKeymaster/IBKeyerBackend.cpp
  ${OFX_SUPPORT_SOURCES}
)

if(WIN32 OR (UNIX AND NOT APPLE))
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 86 89)
  endif()
  if(WIN32)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")
  endif()
  find_package(CUDAToolkit REQUIRED)
  list(APPEND PLUGIN_SOURCES IBKeymaster/IBKeyerCuda.cu)
endif()

if(APPLE)
  enable_language(OBJCXX)
  set(CMAKE_OBJCXX_STANDARD 17)
  set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
  list(APPEND PLUGIN_SOURCES IBKeymaster/MetalKernel.mm)
endif()

add_library(${PLUGIN_BASENAME} SHARED ${PLUGIN_SOURCES})

target_include_directories(${PLUGIN_BASENAME} PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/OpenFX-1.4/include
  ${CMAKE_CURRENT_SOURCE_DIR}/Support/include
  ${CMAKE_CURRENT_SOURCE_DIR}/IBKeymaster
)

target_compile_definitions(${PLUGIN_BASENAME} PRIVATE
  _CRT_SECURE_NO_WARNINGS
)

if(WIN32 OR (UNIX AND NOT APPLE))
  target_compile_definitions(${PLUGIN_BASENAME} PRIVATE
    OFX_SUPPORTS_CUDARENDER
  )
  target_include_directories(${PLUGIN_BASENAME} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
endif()

if(MSVC)
  target_compile_options(${PLUGIN_BASENAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/W3 /EHsc /bigobj /MP>
  )
endif()

if(WIN32)
  target_link_libraries(${PLUGIN_BASENAME} PRIVATE CUDA::cudart)
elseif(UNIX AND NOT APPLE)
  target_link_libraries(${PLUGIN_BASENAME} PRIVATE CUDA::cudart)
  target_link_options(${PLUGIN_BASENAME} PRIVATE -static-libstdc++ -static-libgcc)
endif()

if(APPLE)
  target_link_libraries(${PLUGIN_BASENAME} PRIVATE
    "-framework Metal"
    "-framework Foundation"
    "-framework CoreFoundation"
    "-framework AppKit"
  )
endif()

set_target_properties(${PLUGIN_BASENAME} PROPERTIES
  PREFIX ""
  SUFFIX ".ofx"
  OUTPUT_NAME "${PLUGIN_BASENAME}"
)

if(WIN32)
  set(OFX_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/IBKeymaster/${PLUGIN_BASENAME}.ofx.bundle/Contents/Win64")
elseif(APPLE)
  set(OFX_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/IBKeymaster/${PLUGIN_BASENAME}.ofx.bundle/Contents/MacOS")
else()
  set(OFX_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/IBKeymaster/${PLUGIN_BASENAME}.ofx.bundle/Contents/Linux-x86-64")
endif()

file(MAKE_DIRECTORY "${OFX_OUTPUT_DIR}")

set_target_properties(${PLUGIN_BASENAME} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${OFX_OUTPUT_DIR}"
  RUNTIME_OUTPUT_DIRECTORY_DEBUG "${OFX_OUTPUT_DIR}"
  RUNTIME_OUTPUT_DIRECTORY_RELEASE "${OFX_OUTPUT_DIR}"
  RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${OFX_OUTPUT_DIR}"
  RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${OFX_OUTPUT_DIR}"
  LIBRARY_OUTPUT_DIRECTORY "${OFX_OUTPUT_DIR}"
  LIBRARY_OUTPUT_DIRECTORY_DEBUG "${OFX_OUTPUT_DIR}"
  LIBRARY_OUTPUT_DIRECTORY_RELEASE "${OFX_OUTPUT_DIR}"
  LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO "${OFX_OUTPUT_DIR}"
  LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL "${OFX_OUTPUT_DIR}"
)
