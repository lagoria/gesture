cmake_minimum_required(VERSION 3.5)

project(gesture LANGUAGES CXX VERSION 1.2.1)

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
    WIN32   # 取消控制台窗口
    main.cpp
    mainwindow.cpp
    mainwindow.h
    mainwindow.ui
    inferenceEngine.cpp
    inferenceEngine.h
    ${qrc_FILES}
    eventgroup.h
    eventgroup.cpp

  )
endif()

target_link_libraries(gesture PRIVATE Qt5::Widgets)


# 检查构建生成器
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    # 使用MSVC构建时的设置
    message(STATUS "Using MSVC generator")

    add_compile_options(/utf-8)
    # 为 C 编译器添加 /utf-8 标志
    string(APPEND CMAKE_C_FLAGS " /utf-8")
    # 为 C++ 编译器添加 /utf-8 标志
    string(APPEND CMAKE_CXX_FLAGS " /utf-8")

    # add OpenCV library
    set(OpenCV_DIR "I:/Library/opencv-msvc16-4.7.0")
    find_package(OpenCV REQUIRED)
    message(STATUS "OpenCV version: ${OpenCV_VERSION}")
    target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})

    # add ONNX Runtime library
    set(ONNXRUNTIME_ROOT_PATH "I:/Library/onnxruntime-vc16-1.12.0")
    include_directories(${ONNXRUNTIME_ROOT_PATH}/include)
    link_directories(${ONNXRUNTIME_ROOT_PATH}/lib)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${ONNXRUNTIME_ROOT_PATH}/lib/onnxruntime.lib)

else()
    # 使用MinGW构建时的设置
    add_compile_definitions(USE_MINGW_COMPILER)
    message(STATUS "Using MinGW generator")

    # add OpenCV library
    set(OpenCV_DIR "I:/Library/opencv-mingw-4.8.0")
    find_package(OpenCV REQUIRED)
    message(STATUS "OpenCV version: ${OpenCV_VERSION}")
    target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})

endif()


# 工程版本参数配置
string(TIMESTAMP COMPILE_TIME %Y%m%d-%H%M%S)
configure_file(VersionConfig.h.in VersionConfig.h)
# 添加VersionConfig.h文件
target_include_directories(${PROJECT_NAME} PRIVATE ${PROJECT_BINARY_DIR})




