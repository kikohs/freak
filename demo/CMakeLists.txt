include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}/../src
	${OpenCV_INCLUDE_DIRS}
)

if(BUILD_DEMO)

    add_executable(demo demo.cpp)
    target_link_libraries(demo freak_p
            ${OpenCV_LIBS})

endif(BUILD_DEMO)
