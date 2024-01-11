QT  += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = TestAIDet
TEMPLATE = app
#TRANSLATIONS = temui.ts

DEFINES += QT_DEPRECATED_WARNINGS

INCLUDEPATH +=\
        ../../BaseLibX64/Halxl/include\


LIBS +=\
        ../../BaseLibX64/Halxl/lib/halconcpp.lib
        ../../BaseLibX64/Halxl/lib/halconxl.lib


SOURCES += \
        hdevai.cpp \
        main.cpp \
        #mainwindow.cpp \
        #TrainDetect.cpp

HEADERS += \
    hdevai.h \
        #mainwindow.h \


win32:DESTDIR = ../../BIN

FORMS += \
    #mainwindow.ui
