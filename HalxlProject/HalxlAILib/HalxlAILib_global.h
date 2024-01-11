#ifndef HALXLAILIB_GLOBAL_H
#define HALXLAILIB_GLOBAL_H

#include <QtCore/qglobal.h>

#if defined(HALXAILIB_LIBRARY)
#  define HALXAILIB_EXPORT Q_DECL_EXPORT
#else
#  define HALXAILIB_EXPORT Q_DECL_IMPORT
#endif

#endif // HALXLAILIB_GLOBAL_H
