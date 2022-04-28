cmake_minimum_required(VERSION 3.0)
project(Vulkan)

set(CMAKE_CXX_STANDARD 17)

find_package(Vulkan REQUIRED)

file(GLOB SRC
    "src/*.cpp"
    "include/*.hpp"
)

set(GLFW_BUILD_DOCS OFF)
set(GLFW_BUILD_TESTS OFF)
set(GLFW_BUILD_EXAMPLES OFF)

add_subdirectory(3rdparty)

include_directories(include)

add_executable(${PROJECT_NAME} ${SRC})

target_link_libraries(${PROJECT_NAME} Vulkan::Vulkan)
target_link_libraries(${PROJECT_NAME} glfw)
target_link_libraries(${PROJECT_NAME} glm)