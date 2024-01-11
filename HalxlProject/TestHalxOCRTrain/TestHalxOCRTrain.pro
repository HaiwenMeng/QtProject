QT -= gui

CONFIG += c++11 console
CONFIG -= app_bundle

DEFINES += QT_DEPRECATED_WARNINGS

INCLUDEPATH +=\
        ../../BaseLibX64/Halxl/include\


LIBS +=\
        ../../BaseLibX64/Halxl/lib/halconcpp.lib\
        ../../BaseLibX64/Halxl/lib/halconxl.lib\
        ../../BaseLibX64/Halxl/lib/halconxl.lib\


SOURCES += \
        #halxAI.cpp \
        main.cpp \


HEADERS += \
    #halxAI.h \



win32:DESTDIR = ../../BIN


