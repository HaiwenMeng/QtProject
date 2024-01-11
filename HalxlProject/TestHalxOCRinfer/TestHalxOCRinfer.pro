QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++11

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    main.cpp \
    mainwindow.cpp

HEADERS += \
    mainwindow.h

FORMS += \
    mainwindow.ui

INCLUDEPATH +=\
        ../../BaseLibX64/Halxl/include\

LIBS +=\
        ../../BaseLibX64/Halxl/lib/halconcpp.lib
        ../../BaseLibX64/Halxl/lib/halconxl.lib


win32:DESTDIR = ../../BIN
