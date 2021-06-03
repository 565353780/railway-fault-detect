#-------------------------------------------------
#
# Project created by QtCreator 2020-09-28T17:54:39
#
#-------------------------------------------------

QT       += core gui network widgets

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = RailwayFaultDetect
TEMPLATE = app

win32{
DESTDIR = ../bin_win
}
unix{
DESTDIR = ../bin_linux
}

# The following define makes your compiler emit warnings if you use
# any feature of Qt which has been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

DEFINES -= UNICODE

#DEFINES += Linux

win32{
DEFINES += WIN32 \
           OPENCV \
           GPU
}
unix{
DEFINES += Linux
}

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

CONFIG += c++11

SOURCES += \
    main.cpp \
    RailwayFaultDetectWidget.cpp \
    DarknetDetector.cpp \
    OpencvCap.cpp \
    VideoCap.cpp \
    ShowImageWidget.cpp \
    DataReader.cpp \
    LapnetDetector.cpp \
    HttpApi.cpp \
    PostProcesser.cpp

HEADERS += \
    RailwayFaultDetectWidget.h \
    DarknetDetector.h \
    OpencvCap.h \
    VideoCap.h \
    ShowImageWidget.h \
    DataReader.h \
    LapnetDetector.h \
    HttpApi.h \
    PostProcesser.h

FORMS += \
    railwayfaultdetectwidget.ui \
    showimagewidget.ui

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

# darknet-win-gpu
win32{
INCLUDEPATH += $$PWD/../thirdparty/OpenCV340/opencv/build/include \
               $$PWD/../thirdparty/darknet-gpu

LIBS += $$PWD/../thirdparty/OpenCV340/opencv/build/x64/vc14/lib/opencv_world340.lib \
        $$PWD/../thirdparty/darknet-gpu/yolo_cpp_dll.lib
}

# darknet-linux-gpu
unix{
DEFINES += \
    GPU \
    CUDNN

HEADERS += ../thirdparty/darknet-gpu/darknet.h

LIBS += /usr/local/lib/libopencv_highgui.so \
        /usr/local/lib/libopencv_highgui.so \
        /usr/local/lib/libopencv_imgcodecs.so \
        /usr/local/lib/libopencv_imgproc.so \
        /usr/local/lib/libopencv_core.so \
        /usr/local/lib/libopencv_videoio.so

LIBS += $$PWD/../../include/libdarknet.so \
        /usr/local/cuda/lib64/libcudart.so.10.2 \
        /usr/local/cuda/lib64/libcudnn.so.7 \
        /usr/local/cuda/lib64/libcurand.so.10 #\
        #/usr/local/cuda/lib64/libcublas.so.10.2

INCLUDEPATH += /usr/local/cuda/targets/x86_64-linux/include
}

OTHER_FILES += \
    Python/lapnet/LapNet.py \
    Python/lapnet/create_dataset.py \
    Python/lapnet/test.py \
    Python/lapnet/test_line.py \
    Python/lapnet/test_point.py \

TRANSLATIONS = language.zh_CN.ts
