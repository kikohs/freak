cmake_minimum_required( VERSION 2.8 )

project( freak )
set( PROJECT_VERSION 0.2.0 )

set( freak_LIB_MAJOR_VERSION "0" )
set( freak_LIB_MINOR_VERSION "2" )
set( freak_LIB_PATCH_VERSION "0" )

set( freak_VERSION ${freak_LIB_MAJOR_VERSION}.${freak_LIB_MINOR_VERSION}.${freak_LIB_PATCH_VERSION} )
set( freak_SOVERSION ${freak_LIB_MAJOR_VERSION} )

#### Required
#    FIND_PACKAGE(OpenCV REQUIRED core highgui)
find_package(OpenCV REQUIRED)

include_directories( ${CMAKE_CURRENT_SOURCE_DIR} )
include_directories( ${CMAKE_CURRENT_BINARY_DIR} )

set( LIB_SUFFIX "" CACHE STRING "Define suffix of directory name (32/64)" )
set( EXEC_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX} CACHE PATH "Installation prefix for executables and object code libraries" FORCE )
set( BIN_INSTALL_DIR ${EXEC_INSTALL_PREFIX}/bin CACHE PATH "Installation prefix for user executables" FORCE )
set( LIB_INSTALL_DIR ${EXEC_INSTALL_PREFIX}/lib${LIB_SUFFIX} CACHE PATH  "Installation prefix for object code libraries" FORCE )
set( INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/include CACHE PATH "Installation prefix for C header files" FORCE )

option( BUILD_SHARED_LIBS "Build shared libraries" ON )
option( BUILD_DEMO "Build demo" ON )
option( USE_SSE "Use SSE3" ON )

if( USE_SSE )
    add_definitions(
        -DCV_SSSE3
    )
endif()

if( CMAKE_COMPILER_IS_GNUCXX )

    if( CMAKE_BUILD_TYPE STREQUAL "Release" )
        set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2" )
    endif()

    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -mssse3" )
    set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -mssse3" )
endif()
if( CMAKE_COMPILER_IS_GNUCXX AND BUILD_SHARED_LIBS )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" )
    set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS}" )
endif()

set( CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules )

#### Config .h
#configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/config.h.in ${CMAKE_CURRENT_BINARY_DIR}/config.h )

# Process subdir
add_subdirectory( src )

if( BUILD_DEMO )
    add_subdirectory( demo )
endif( BUILD_DEMO )

