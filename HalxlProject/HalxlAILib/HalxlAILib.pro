QT -= gui

TARGET = HalxlAILib


CONFIG += dll
TEMPLATE = lib
DEFINES += HALXAILIB_LIBRARY



CONFIG += c++11

CONFIG += staticlib


SOURCES += \
    halxlailib.cpp

HEADERS += \
    HalxlAILib_global.h \
    halxlailib.h


INCLUDEPATH +=\
        ../../BaseLibX64/Halxl/include\

LIBS +=\
        ../../BaseLibX64/Halxl/lib/halconcpp.lib
        ../../BaseLibX64/Halxl/lib/halconxl.lib


DESTDIR = ../../BIN
