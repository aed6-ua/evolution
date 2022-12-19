cmake_minimum_required(VERSION 3.15)
project(evolution)

add_subdirectory(utils)
add_subdirectory(mlp)
add_subdirectory(genetic)

include_directories(${PROJECT_SOURCE_DIR})
include_directories( ${OPENGL_INCLUDE_DIRS}  ${GLUT_INCLUDE_DIRS} )
include_directories(${MPI_INCLUDE_PATH})
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
add_executable(${PROJECT_NAME} main.cc)
target_link_libraries(${PROJECT_NAME} PUBLIC MPI::MPI_CXX ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} utils mlp genetic)
