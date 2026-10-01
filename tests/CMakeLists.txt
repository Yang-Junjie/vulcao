add_executable(vulcao_tests
    acceleration_structure_test.cpp
    barrier_test.cpp
    context_test.cpp
    deletion_queue_test.cpp
    descriptor_test.cpp
    headless_test.cpp
    image_test.cpp
    log_test.cpp
    pipeline_test.cpp
    reflection_test.cpp
    rendering_test.cpp
    resource_test.cpp
    specialization_test.cpp
    transfer_test.cpp
    vertex_layout_test.cpp
)

target_include_directories(vulcao_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(vulcao_tests PRIVATE
    vulcao
    doctest::doctest_with_main
)

# The cases that need SPIR-V are only built when slangc is available, so the suite
# still configures on machines that only ship the Vulkan headers and loader.
include("${VULCAO_CMAKE_DIR}/vulcaoShaders.cmake")
vulcao_find_slangc(SLANGC_EXECUTABLE)
if(SLANGC_EXECUTABLE)
    vulcao_add_slang_shaders(vulcao_tests
        "reflection.slang|vertMain|vertex|reflection.vert.spv"
        "reflection.slang|vertexIndexMain|vertex|reflection.index.vert.spv"
        "reflection.slang|fragMain|fragment|reflection.frag.spv"
        "reflection.slang|computeMain|compute|reflection.comp.spv"
    )
else()
    message(STATUS "vulcao: slangc not found, shader dependent tests are not built")
endif()

add_test(NAME vulcao_tests COMMAND vulcao_tests)
