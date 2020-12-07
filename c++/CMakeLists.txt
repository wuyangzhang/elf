cmake_minimum_required(VERSION 3.12)
project(ElfC)

set(CMAKE_CXX_STANDARD 17)

set(OpenCV_DIR "/usr/local/Cellar/opencv/4.3.0_5/lib/cmake/opencv4")

set(CMAKE_PREFIX_PATH "/home/wuyang/libtorch")

# add torch lib
find_package(Torch REQUIRED)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

add_executable(Elf main.cpp config/Config.h dataset/VideoLoader.cpp dataset/VideoLoader.h run/Client.cpp run/Client.h network/Socket.cpp network/Socket.h test/test_network.cpp test/test.h app/ApplicationInterface.cpp app/ApplicationInterface.h partitioning/PartitionManager.cpp partitioning/PartitionManager.h prediction/PredictionManager.cpp prediction/PredictionManager.h control/ControlManager.cpp control/ControlManager.h test/test_img_transfer.cpp utils/utils.h utils/utils.cpp partitioning/PartitionSimulator.cpp partitioning/PartitionSimulator.h test/test_parallel_offload.cpp test/test_wait_notify.cpp test/test_multi_sockets.cpp run/Elf.cpp run/Elf.h test/test_elf.cpp test/test_array.cpp test/test_predictiion.cpp)

# add libs you need
set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs)

find_package(cppzmq)

# linking
target_link_libraries(Elf ${TORCH_LIBRARIES} cppzmq ${OpenCV_LIBS}  pthread)