cmake_minimum_required(VERSION 3.1)
set (CMAKE_CXX_STANDARD 11)
project( CVTest )
find_package( OpenCV REQUIRED )
add_executable( CVTest main.cpp )
target_link_libraries( CVTest ${OpenCV_LIBS})

set(EXE CVTest)
install (TARGETS ${EXE}
  DESTINATION "/usr/bin")

