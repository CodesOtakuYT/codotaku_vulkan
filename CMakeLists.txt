cmake_minimum_required(VERSION 3.10.0)
project(
  codotaku_vulkan
  VERSION 0.1.0
  LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_subdirectory(vendor/SDL)
add_subdirectory(vendor/glm)
add_subdirectory(vendor/Vulkan-Headers)
add_subdirectory(vendor/VulkanMemoryAllocator)

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS src/*.cpp)

add_executable(${PROJECT_NAME})
target_sources(${PROJECT_NAME} PRIVATE ${SOURCES})
target_link_libraries(
  ${PROJECT_NAME} PRIVATE Vulkan::Headers GPUOpen::VulkanMemoryAllocator
                          SDL3::SDL3 glm::glm)
target_compile_definitions(
  ${PROJECT_NAME} PRIVATE VULKAN_HPP_ENABLE_DYNAMIC_LOADER_TOOL=0
                          SDL_MAIN_USE_CALLBACKS VK_NO_PROTOTYPES)
