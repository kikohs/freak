include_directories( ${CMAKE_CURRENT_SOURCE_DIR} )

# ADD all sources files
set( freak_SRCS
    freak.cpp
)

# Static version
add_library( freak_p STATIC ${freak_SRCS} )
target_link_libraries( freak_p ${OpenCV_LIBS} )

#########
# Shared library

# Set public header
set( freak_HDRS
    freak.h
    hammingseg.h
)

add_library( freak SHARED ${freak_SRCS} ${freak_HDRS} )
set_target_properties( freak PROPERTIES
    VERSION ${freak_VERSION}
    SOVERSION ${freak_SOVERSION}
    INSTALL_NAME_DIR ${LIB_INSTALL_DIR}
    DEFINE_SYMBOL CV_EXPORT
)

target_link_libraries( freak ${OpenCV_LIBS} )

install( TARGETS freak
    FRAMEWORK DESTINATION ${FRAMEWORK_INSTALL_DIR}
    LIBRARY DESTINATION ${LIB_INSTALL_DIR}
    RUNTIME DESTINATION ${BIN_INSTALL_DIR}
    ARCHIVE DESTINATION ${LIB_INSTALL_DIR}
)

install( FILES ${freak_HDRS} DESTINATION ${INCLUDE_INSTALL_DIR}/freak )
