# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

project(atvc_test)
cmake_minimum_required(VERSION 3.16.0)

set(CMAKE_CXX_STANDARD 17)

set(PRODUCT_TYPE_LIST ascend910b
        CACHE STRING "Supported product types"
)

add_compile_definitions(__CCE_KT_TEST__=1 ASCENDC_CPU_DEBUG=1 ASCENDC_DUMP=0)

include(${ATVC_DIR}/cmake/intf_pub_llt.cmake)
include(${ATVC_DIR}/cmake/third_party/gtest.cmake)

set(ASCENDC_SYS_DIR ${ASCEND_DIR}/${CMAKE_SYSTEM_PROCESSOR}-linux/)

set(ASCEND_DEP_LIB_DIR
        ${ASCEND_DIR}/lib64/
        ${ASCEND_DIR}/tools/tikicpulib/lib/
)

set(ATVC_TEST_HEADER_FILES
        ${UT_PATH}/common
        ${ATVC_DIR}
        ${ATVC_DIR}/include
        ${GTEST_INCLUDE}
        ${C_SEC_INCLUDE}
        ${ASCENDC_SYS_DIR}/asc/
        ${ASCENDC_SYS_DIR}/asc/include
        ${ASCENDC_SYS_DIR}/asc/include/adv_api
        ${ASCENDC_SYS_DIR}/asc/include/basic_api
        ${ASCENDC_SYS_DIR}/asc/impl
        ${ASCENDC_SYS_DIR}/asc/impl/basic_api
        ${ASCEND_DIR}/tools/tikicpulib/lib/include
)

set(ATVC_TEST_ascend910b_HEADER_FILES
        ${ATVC_DIR}
)

set(ATVC_TEST_ascend910b_CASE_SRC_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/block/test_add.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/block/test_broadcast_to.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/block/test_reduce_sum.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/block/test_add_with_broadcast.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/block/test_addcmul.cpp
)

set(ATVC_TEST_ascend910b_TILING_CASE_SRC_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/tiling/test_atvc_tiling.cpp
)
# 仅针对ATVC产品类型构建和运行测试
foreach(product_type ${PRODUCT_TYPE_LIST})
    add_executable(atvc_${product_type}_ut
            ${UT_PATH}/main_global.cpp
            $<$<STREQUAL:${product_type},ascend910b>:${UT_PATH}/common/platform_stub.cpp>
            ${ATVC_TEST_${product_type}_CASE_SRC_FILES}
            ${ATVC_TEST_${product_type}_TILING_CASE_SRC_FILES}
    )

    # add soc version flags
    if(${product_type} STREQUAL "ascend910b")
        target_compile_definitions(atvc_${product_type}_ut PRIVATE
                UT_TEST
                __CCE_AICORE__=220
                __NPU_ARCH__=2201
        )
        target_link_directories(atvc_${product_type}_ut PRIVATE
                ${ASCEND_DEP_LIB_DIR}
                ${ASCEND_DIR}/tools/tikicpulib/lib/Ascend910B1/
                ${ASCENDC_SYS_DIR}/simulator/Ascend910B1/lib/
        )
    endif()

    target_compile_definitions(atvc_${product_type}_ut PRIVATE
            __CCE_KT_TEST__=1
            ASCENDC_CPU_DEBUG=1
            ASCENDC_OOM=1
    )

    target_include_directories(atvc_${product_type}_ut PRIVATE
            ${ATVC_TEST_HEADER_FILES}
            ${ATVC_TEST_${product_type}_HEADER_FILES}
    )

    target_compile_options(atvc_${product_type}_ut PRIVATE
            -fno-access-control
    )

    target_link_libraries(atvc_${product_type}_ut PRIVATE
            $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
            $<BUILD_INTERFACE:dlog_headers>
            -Wl,--no-as-needed
            gtest
            gcov
            tikicpulib_npuchk
            tikicpulib_cceprint
            tikicpulib_stubreg
            tikcpp_debug
            $<$<STREQUAL:${product_type},ascend910b>:pem_davinci>
            c_sec
            error_manager
            -Wl,--as-needed
    )

    if(ENABLE_UT_EXEC)
        if(ENABLE_ASAN)
            execute_process(
                    COMMAND ${CMAKE_C_COMPILER} -print-file-name=libasan.so
                    OUTPUT_VARIABLE LIBASAN_PATH
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    RESULT_VARIABLE result
            )
            if(NOT result EQUAL 0)
                message(FATAL_ERROR "compiler not support asan, please disable asan")
            endif()

            execute_process(
                    COMMAND ${CMAKE_C_COMPILER} -print-file-name=libstdc++.so
                    OUTPUT_VARIABLE LIBSTDC_PATH
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    RESULT_VARIABLE stdc_result
            )
            if(NOT stdc_result EQUAL 0)
                message(FATAL_ERROR "not find libstdc++.so, please confirm.")
            endif()

            set(PRELOAD "LD_PRELOAD=${LIBASAN_PATH}:${LIBSTDC_PATH}")
            add_custom_command(
                    TARGET atvc_${product_type}_ut POST_BUILD
                    COMMAND
                    COMMAND bash -c "source ${ASCEND_DIR}/set_env.sh && \
        ${PRELOAD} ASAN_OPTIONS=detect_leaks=0 ./atvc_${product_type}_ut"
                    VERBATIM
                    COMMENT "Run atvc utest with asan"
            )
        else()
            add_custom_command(
                    TARGET atvc_${product_type}_ut POST_BUILD
                    COMMAND bash -c "source ${ASCEND_DIR}/set_env.sh && ./atvc_${product_type}_ut"
                    VERBATIM
                    COMMENT "Run atvc utest"
            )
        endif()

        if(ENABLE_COVERAGE)
            # set(_collect_coverage_data_target collect_coverage_data)
            get_filename_component(_atvc_builtin_bin_path ${CMAKE_BINARY_DIR} DIRECTORY)
            set(_cov_report ${CMAKE_BINARY_DIR}/cov_report)
            set(_cov_html ${_cov_report})
            set(_cov_data ${_cov_report}/coverage.info)
            set(_generate_cov_script ${UT_PATH}/scripts/generate_cpp_cov.sh)

            add_custom_command(
                    TARGET atvc_${product_type}_ut POST_BUILD
                    COMMAND bash ${_generate_cov_script} ${_atvc_builtin_bin_path} ${_cov_data} ${_cov_html}
                    COMMENT "Run collect coverage data"
            )
        endif()
    endif()
endforeach()
