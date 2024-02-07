cmake_minimum_required(VERSION 3.5)

project(gesture LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# QtCreator supports the following variables for Android, which are identical to qmake Android variables.
# Check http://doc.qt.io/qt-5/deployment-android.html for more information.
# They need to be set before the find_package(Qt5 ...) call.

#if(ANDROID)
#    set(ANDROID_PACKAGE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/android")
#    if (ANDROID_ABI STREQUAL "armeabi-v7a")
#        set(ANDROID_EXTRA_LIBS
#            ${CMAKE_CURRENT_SOURCE_DIR}/path/to/libcrypto.so
#            ${CMAKE_CURRENT_SOURCE_DIR}/path/to/libssl.so)
#    endif()
#endif()

find_package(Qt5 COMPONENTS Widgets REQUIRED)

qt5_add_resources(qrc_FILES res.qrc)

if(ANDROID)
  add_library(gesture SHARED
    main.cpp
    mainwindow.cpp
    mainwindow.h
    mainwindow.ui
    yolov5.cpp
    yolov5.h
    ${qrc_FILES}
  )
else()
  add_executable(gesture
    main.cpp
    mainwindow.cpp
    mainwindow.h
    mainwindow.ui
    yolov5.cpp
    yolov5.h
    ${qrc_FILES}
    eventgroup.h eventgroup.cpp
  )
endif()

target_link_libraries(gesture PRIVATE Qt5::Widgets)

# add OpenCV library
set(OpenCV_DIR "I:/OpenCV/build")
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})

# add OpenVINO library
set(OpenVINO_DIR "C:/Program Files (x86)/Intel/openvino_2023.3.0/runtime/cmake")
find_package(OpenVINO REQUIRED)
message(STATUS "OpenVINO version: ${OpenVINO_VERSION}")
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenVINO_LIBS})

