/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <string>
#include <map>
#include "tiling/platform/platform_ascendc.h"
#if __CCE_AICORE__ == 300
static const char *g_core_num_aic = "1";
static const char *g_core_num_aiv = "1";
static const char *g_core_num_cub = "1";
static const char *g_core_type_list = "AICore,CubeCore,VectorCore";
static const char *g_chip_version = "Ascend310B";
static uint32_t g_core_num = 1;
#elif __CCE_AICORE__ == 220
static const char *g_core_num_aic = "24";
static const char *g_core_num_aiv = "48";
static const char *g_core_num_cub = "24";
static const char *g_core_type_list = "CubeCore,VectorCore";
static const char *g_chip_version = "Ascend910B";
static uint32_t g_core_num = 48;
#elif __CCE_AICORE__ == 200
static const char *g_core_num_aic = "10";
static const char *g_core_num_aiv = "8";
static const char *g_core_num_cub = "10";
static const char *g_core_type_list = "AICore,VectorCore";
static const char *g_chip_version = "Ascend310P";
static uint32_t g_core_num = 10;
#elif defined(__DAV_C310__) || defined(__DAV_310R6__) || (__NPU_ARCH__ == 5102)
static const char *g_core_num_aic = "24";
static const char *g_core_num_aiv = "48";
static const char *g_core_num_cub = "24";
static const char *g_core_type_list = "AICore,VectorCore";
static const char *g_chip_version = "Ascend910_95";
static uint32_t g_core_num = 48;
#else
static const char *g_core_num_aic = "32";
static const char *g_core_num_aiv = "0";
static const char *g_core_num_cub = "32";
static const char *g_core_type_list = "AICore";
static const char *g_chip_version = "Ascend910";
static uint32_t g_core_num = 32;
#endif


void platfrom_stub_set_num_aic(const char *num)
{
    g_core_num_aic = num;
}

void platfrom_stub_set_num_aiv(const char *num)
{
    g_core_num_aiv = num;
}

void platfrom_stub_set_num_cub(const char *num)
{
    g_core_num_cub = num;
}

void platfrom_stub_set_ctl(const char *num)
{
    g_core_type_list = num;
}

void platfrom_stub_set_chip_version(const char *num)
{
    g_chip_version = num;
}

void platfrom_stub_set_num(uint32_t num)
{
    g_core_num = num;
}

namespace fe {
enum class LocalMemType {
  L0_A = 0,
  L0_B = 1,
  L0_C = 2,
  L1 = 3,
  L2 = 4,
  UB = 5,
  HBM = 6,
  RESERVED
};
class OptionalInfos {
 public:
  bool Init();

};

class PlatFormInfos {
public:
uint32_t GetCoreNum(void) const;
uint32_t GetCoreNumAiv(void) const;
bool GetPlatformRes(const std::string &label, const std::string &key, std::string &val);
bool GetPlatformResWithLock(const std::string &label, const std::string &key, std::string &val);
void GetLocalMemSize(const LocalMemType &mem_type, uint64_t &size);
void GetLocalMemBw(const LocalMemType &mem_type, uint64_t &bw_size);
void SetCoreNumByCoreType(const std::string &core_type);
};
void PlatFormInfos::SetCoreNumByCoreType(const std::string &core_type)
{
    return;
}
uint32_t PlatFormInfos::GetCoreNum(void) const
{
    return g_core_num;
}
uint32_t PlatFormInfos::GetCoreNumAiv(void) const
{
    return g_core_num;
}
bool PlatFormInfos::GetPlatformRes(const std::string &label, const std::string &key, std::string &val)
{
    if (label.compare("SoCInfo") != 0 && label.compare("version") != 0) {
        val = "0";
        return false;
    }

    if (key.compare("core_type_list") == 0) {
        val = g_core_type_list;
    } else if (key.compare("ai_core_cnt") == 0) {
        val = g_core_num_aic;
    } else if (key.compare("cube_core_cnt") == 0) {
        val = g_core_num_cub;
    } else if (key.compare("vector_core_cnt") == 0) {
        val = g_core_num_aiv;
    } else if (key.compare("Short_SoC_version") == 0) {
        val = g_chip_version;
    } else {
        val = "0";
        return false;
    }

    return true;
}

bool PlatFormInfos::GetPlatformResWithLock(const std::string &label, const std::string &key, std::string &val)
{
    return GetPlatformRes(label, key, val);
}

#if __CCE_AICORE__ == 300
void PlatFormInfos::GetLocalMemSize(const LocalMemType &mem_type, uint64_t &size)
{
    switch(mem_type) {
        case LocalMemType::L0_A:
            size = 65536;
            break;
        case LocalMemType::L0_B:
            size = 65536;
            break;
        case LocalMemType::L0_C:
            size = 131072;
            break;
        case LocalMemType::UB:
            size = 253952;
            break;
        case LocalMemType::L1:
            size = 1048576;
            break;
        case LocalMemType::L2:
            size = 4194304;
            break;
        case LocalMemType::HBM:
            size = 0;
            break;
        default:
            size = 0;
            break;
    }
}
void PlatFormInfos::GetLocalMemBw(const LocalMemType &mem_type, uint64_t &bw_size)
{
    switch(mem_type) {
        case LocalMemType::L2:
            bw_size = 256;
            break;
        case LocalMemType::HBM:
            bw_size = 17;
            break;
        default:
            bw_size = 0;
            break;
    }

}
#elif __CCE_AICORE__ == 220
void PlatFormInfos::GetLocalMemSize(const LocalMemType &mem_type, uint64_t &size)
{
    switch(mem_type) {
        case LocalMemType::L0_A:
            size = 65536;
            break;
        case LocalMemType::L0_B:
            size = 65536;
            break;
        case LocalMemType::L0_C:
            size = 131072;
            break;
        case LocalMemType::UB:
            size = 196608;
            break;
        case LocalMemType::L1:
            size = 524288;
            break;
        case LocalMemType::L2:
            size = 201326592;
            break;
        case LocalMemType::HBM:
            size = 0;
            break;
        default:
            size = 0;
            break;
    }
}
void PlatFormInfos::GetLocalMemBw(const LocalMemType &mem_type, uint64_t &bw_size)
{
    switch(mem_type) {
        case LocalMemType::L2:
            bw_size = 110;
            break;
        case LocalMemType::HBM:
            bw_size = 32;
            break;
        default:
            bw_size = 0;
            break;
    }

}
#elif __CCE_AICORE__ == 200
void PlatFormInfos::GetLocalMemSize(const LocalMemType &mem_type, uint64_t &size)
{
    switch(mem_type) {
        case LocalMemType::L0_A:
            size = 65536;
            break;
        case LocalMemType::L0_B:
            size = 65536;
            break;
        case LocalMemType::L0_C:
            size = 262144;
            break;
        case LocalMemType::UB:
            size = 262144;
            break;
        case LocalMemType::L1:
            size = 1048576;
            break;
        case LocalMemType::L2:
            size = 16777216;
            break;
        case LocalMemType::HBM:
            size = 0;
            break;
        default:
            size = 0;
            break;
    }
}
void PlatFormInfos::GetLocalMemBw(const LocalMemType &mem_type, uint64_t &bw_size)
{
    switch(mem_type) {
        case LocalMemType::L2:
            bw_size = 114;
            break;
        case LocalMemType::HBM:
            bw_size = 17;
            break;
        default:
            bw_size = 0;
            break;
    }

}
#else
void PlatFormInfos::GetLocalMemSize(const LocalMemType &mem_type, uint64_t &size)
{
    switch(mem_type) {
        case LocalMemType::L0_A:
            size = 65536;
            break;
        case LocalMemType::L0_B:
            size = 65536;
            break;
        case LocalMemType::L0_C:
            size = 262144;
            break;
        case LocalMemType::UB:
            size = 262144;
            break;
        case LocalMemType::L1:
            size = 1048576;
            break;
        case LocalMemType::L2:
            size = 33554432;
            break;
        case LocalMemType::HBM:
            size = 0;
            break;
        default:
            size = 0;
            break;
    }
}
void PlatFormInfos::GetLocalMemBw(const LocalMemType &mem_type, uint64_t &bw_size)
{
    switch(mem_type) {
        case LocalMemType::L2:
            bw_size = 110;
            break;
        case LocalMemType::HBM:
            bw_size = 32;
            break;
        default:
            bw_size = 0;
            break;
    }

}
#endif
class PlatformInfoManager{
public:
    static PlatformInfoManager &GeInstance();
    uint32_t InitializePlatformInfo();
    uint32_t GetPlatformInfos(const std::string SoCVersion,
                            PlatFormInfos &platform_info,
                            OptionalInfos &opti_compilation_info);
    uint32_t InitRuntimePlatformInfos(const std::string &SoCVersion);
};
uint32_t PlatformInfoManager::InitRuntimePlatformInfos(const std::string &soCVersion)
{
    return 0;
}
uint32_t PlatformInfoManager::GetPlatformInfos(const std::string SoCVersion,
                            PlatFormInfos &platform_info,
                            OptionalInfos &opti_compilation_info) {
    return 0;
}
PlatformInfoManager& PlatformInfoManager::GeInstance() {
  static PlatformInfoManager pf;
  return pf;
}
}
namespace platform_ascendc {
PlatformAscendC* PlatformAscendCManager::platformInfo = nullptr;
std::mutex PlatformAscendCManager::platformInitMtx;
SocVersion PlatformAscendCManager::SocVersionMap(const char *socVersionStr)
{
    return SocVersion::ASCEND910B;
}
fe::PlatFormInfos* PlatformAscendCManager::PlatformAscendCInit(const char *customSocVersion)
{
    return nullptr;
}
PlatformAscendC* PlatformAscendCManager::PlatformAscendCManagerInit(const char *customSocVersion)
{
    fe::PlatFormInfos pfs;
    static PlatformAscendC pfc(&pfs);
    platformInfo = &pfc;
    return nullptr;
}
SocVersion PlatformAscendC::GetSocVersion(void) const {
    std::string socVersionStr;
    const auto ret = this->platformInfo_->GetPlatformResWithLock("version", "Short_SoC_version", socVersionStr);
    if (!ret) {
        return SocVersion::RESERVED_VERSION;
    }
    static std::map<std::string, SocVersion> convertMap = {
        {"Ascend310P", SocVersion::ASCEND310P},
        {"Ascend910", SocVersion::ASCEND910},
        {"Ascend910B", SocVersion::ASCEND910B},
        {"Ascend910_93", SocVersion::ASCEND910B},
        {"Ascend910_95", SocVersion::ASCEND910_95},
        {"Ascend310B", SocVersion::ASCEND310B},
        {"MC62CM12A", SocVersion::MC62CM12A},
    };
    auto it = convertMap.find(socVersionStr);
    if (it != convertMap.end()) {
        return it->second;
    }
    return SocVersion::RESERVED_VERSION;
}
void PlatformAscendC::GetCoreMemSize(const CoreMemType &memType, uint64_t &size) const {
    const fe::LocalMemType localType = static_cast<fe::LocalMemType>(memType);
    this->platformInfo_->GetLocalMemSize(localType, size);
    // only ascend910B need UB/L1 local reserved buf for kfc
    if ((memType == CoreMemType::UB || memType == CoreMemType::L1)
         && GetSocVersion() == SocVersion::ASCEND910B) {
        size -= 256;
    }
}

uint32_t PlatformAscendC::GetCoreNumAiv(void) const {
    return this->platformInfo_->GetCoreNumAiv();
}
}
