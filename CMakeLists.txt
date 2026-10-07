cmake_minimum_required(VERSION 3.24)
project(vulcao VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(VULCAO_BUILD_TESTS "Build vulcao tests" ${PROJECT_IS_TOP_LEVEL})
option(VULCAO_BUILD_DOCS "Build API documentation with Doxygen" OFF)
option(VULCAO_INSTALL "Generate vulcao install rules" ${PROJECT_IS_TOP_LEVEL})

set(VULCAO_THIRD_PARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
set(VULCAO_CMAKE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(Vulkan REQUIRED)

# --- vulcao dependencies -----------------------------------------------------

set(VK_BOOTSTRAP_TEST OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/vk-bootstrap)

set(VMA_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/VulkanMemoryAllocator)

set(SPIRV_REFLECT_EXECUTABLE OFF CACHE BOOL "" FORCE)
set(SPIRV_REFLECT_STATIC_LIB ON CACHE BOOL "" FORCE)
set(SPIRV_REFLECT_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/spirv-reflect)

# --- vulcao ------------------------------------------------------------------

add_library(vulcao STATIC
    src/acceleration_structure.cpp
    src/allocator.cpp
    src/barrier.cpp
    src/buffer.cpp
    src/command_buffer.cpp
    src/command_pool.cpp
    src/context.cpp
    src/descriptor_set.cpp
    src/event.cpp
    src/fence.cpp
    src/frame_manager.cpp
    src/image.cpp
    src/image_view.cpp
    src/log.cpp
    src/pipeline.cpp
    src/pipeline_cache.cpp
    src/pipeline_layout.cpp
    src/query_pool.cpp
    src/reflection.cpp
    src/rendering.cpp
    src/sampler.cpp
    src/semaphore.cpp
    src/shader_module.cpp
    src/surface.cpp
    src/swapchain.cpp
    src/vma_impl.cpp
    src/window_presenter.cpp
)

target_include_directories(vulcao PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:include>"
)

target_compile_features(vulcao PUBLIC cxx_std_20)

target_link_libraries(vulcao PUBLIC
    Vulkan::Vulkan
    Vulkan::Headers
    "$<BUILD_INTERFACE:vk-bootstrap::vk-bootstrap>"
    "$<INSTALL_INTERFACE:vulcao::vk-bootstrap>"
    "$<BUILD_INTERFACE:VulkanMemoryAllocator::Headers>"
    "$<INSTALL_INTERFACE:vulcao::vma>"
    "$<BUILD_INTERFACE:spirv-reflect-static>"
    "$<INSTALL_INTERFACE:vulcao::spirv-reflect>"
)

target_compile_definitions(vulcao PUBLIC
    VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
)

if(VULCAO_INSTALL)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)

    install(TARGETS vulcao
        EXPORT vulcaoTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

    install(DIRECTORY include/vulcao DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

    install(TARGETS vk-bootstrap spirv-reflect-static
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

    install(FILES
        "${VULCAO_THIRD_PARTY_DIR}/vk-bootstrap/src/VkBootstrap.h"
        "${VULCAO_THIRD_PARTY_DIR}/spirv-reflect/spirv_reflect.h"
        "${VULCAO_THIRD_PARTY_DIR}/VulkanMemoryAllocator/include/vk_mem_alloc.h"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/vulcao/third_party")

    install(EXPORT vulcaoTargets
        FILE vulcaoTargets.cmake
        NAMESPACE vulcao::
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/vulcao")

    configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/vulcaoConfig.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/vulcaoConfig.cmake"
        INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/vulcao")

    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/vulcaoConfigVersion.cmake"
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion)

    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/vulcaoConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/vulcaoConfigVersion.cmake"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/vulcao")
endif()

# --- tests -------------------------------------------------------------------

if(VULCAO_BUILD_TESTS)
    enable_testing()

    set(DOCTEST_WITH_TESTS OFF CACHE BOOL "" FORCE)
    set(DOCTEST_WITH_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(DOCTEST_NO_INSTALL ON CACHE BOOL "" FORCE)
    add_subdirectory(third_party/doctest)

    add_subdirectory(tests)
endif()

# --- documentation -----------------------------------------------------------

if(VULCAO_BUILD_DOCS)
    find_package(Doxygen REQUIRED)

    add_custom_target(vulcao_docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM)
endif()
