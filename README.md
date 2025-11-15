# ATVC

## 🔥Latest News

- [2025/11] ATVC项目首次上线。

## 🚀概述

ATVC（Ascend C Template for Vector Compute）是一个用Ascend C API搭建的C++模板头文件集合，旨在帮助用户快速开发Ascend C典型Vector算子。它将Ascend C Vector算子开发流程中的计算实现解耦成可自定义的模块，内部封装实现了Kernel数据搬入搬出等底层通用操作及通用Tiling计算，实现了高效的算子开发模式。
相比传统Ascend C算子开发方式，利用ATVC搭建的Vector算子可做到开发效率提升3-5倍。用户只需选择匹配的模板并完成核心计算逻辑就完成算子Kernel侧开发，ATVC还内置了每个模板库对应的通用Tiling计算实现，可省去用户手写Tiling的开发量就能达到不错的性能表现，极大提升算子开发效率。

## 🔍目录结构

ATVC代码目录结构如下：

```
├── build.sh                            # 项目工程编译脚本
├── cmake                               # 项目工程编译目录
├── CMakeLists.txt                      # 编译配置文件
├── docs                                # 项目文档介绍
├── examples                            # ATVC 样例
├── include                             # 项目公共头文件
├── README.md
├── scripts                             # 项目脚本文件存放目录
├── test                                # UT测试工程目录
```

## ⚡️快速入门

若您希望快速体验项目，请访问[快速入门](./docs/01_quick_start.md)获取简易教程，包括环境搭建、编译执行、本地验证等操作。

- [环境准备](./docs/01_quick_start.md#环境准备)：安装软件包之前，需要完成搭建基础环境，包括第三方依赖等；基础环境搭建后需要完成社区版CANN软件包安装、环境变量配置等。
- [源码下载](./docs/01_quick_start.md#源码下载)：本项目源码下载。
- [编译安装](./docs/01_quick_start.md#编译安装)：环境准备好后，可对源码修改编译生成可部署的安装包。
- [UT测试](./docs/01_quick_start.md#ut测试(可选))：基于项目根目录的build.sh脚本，可执行UT用例，快速验证功能。
- [样例运行验证](./docs/01_quick_start.md#样例运行验证)：基础样例的编译、执行。

## 📖文档介绍

| 文档 | 说明 |
|------|------|
|[快速入门](./docs/01_quick_start.md)|快速体验项目的简易教程。|
|[编程指南](./docs/02_developer_guide.md)|使用ATVC实现算子开发的教程。|

## 📝相关信息

- [贡献指南](CONTRIBUTING.md)
- [安全声明](SECURITY.md)
- [许可证](LICENSE)
