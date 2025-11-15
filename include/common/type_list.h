/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATVC_TYPE_LIST_H
#define ATVC_TYPE_LIST_H

#include <cstddef>

namespace ATVC {
template <size_t N>
struct SizeValue {
    static const size_t VALUE = N;
};

template <typename Acc, typename T>
struct SumSizes {
    using Type = SizeValue<Acc::VALUE + sizeof(T)>;
};

template <typename... Ts>
struct TypeList {};

template <typename List>
struct TypeListSize {};

template <typename... Ts>
struct TypeListSize<TypeList<Ts...>> {
    static constexpr size_t VALUE = sizeof...(Ts);
};

template <typename TypeList, std::size_t N>
struct TypeListGet {};

template <typename T, typename... Ts, std::size_t N>
struct TypeListGet<TypeList<T, Ts...>, N> {
    using Type = typename TypeListGet<TypeList<Ts...>, N - 1>::Type;
};

template <typename T, typename... Ts>
struct TypeListGet<TypeList<T, Ts...>, 0> {
    using Type = T;
};

template <std::size_t N>
struct TypeListGet<TypeList<>, N> {
};

template <typename TypeList, std::size_t N>
struct TypeListByteOffset {};

template <std::size_t N>
struct TypeListByteOffset<TypeList<>, N> {
    static constexpr std::size_t VALUE = 0;
};

template <typename Head, typename... Tail>
struct TypeListByteOffset<TypeList<Head, Tail...>, 0> {
    static constexpr std::size_t VALUE = 0;
};

template <typename Head, typename... Tail, std::size_t N>
struct TypeListByteOffset<TypeList<Head, Tail...>, N> {
    static constexpr std::size_t VALUE = sizeof(Head) + TypeListByteOffset<TypeList<Tail...>, N - 1>::VALUE;
};

template <typename T, typename List>
struct TypeListPrepend {};

template <typename T, typename... Ts>
struct TypeListPrepend<T, TypeList<Ts...>> {
    using Type = TypeList<T, Ts...>;
};

template <typename List, template <typename> class Mapper>
struct TypeListMap {};

template <template <typename> class Mapper>
struct TypeListMap<TypeList<>, Mapper> {
    using Type = TypeList<>;
};

template <typename Head, typename... Tail, template <typename> class Mapper>
struct TypeListMap<TypeList<Head, Tail...>, Mapper> {
private:
    using MappedHead = typename Mapper<Head>::Type;
    using MappedTail = typename TypeListMap<TypeList<Tail...>, Mapper>::Type;

public:
    using Type = typename TypeListPrepend<MappedHead, MappedTail>::Type;
};

template <typename List, typename Init, template <typename, typename> class Reducer>
struct TypeListReduce {};

template <typename Init, template <typename, typename> class Reducer>
struct TypeListReduce<TypeList<>, Init, Reducer> {
    using Type = Init;
};

template <typename Head, typename... Tail, typename Init, template <typename, typename> class Reducer>
struct TypeListReduce<TypeList<Head, Tail...>, Init, Reducer> {
private:
    using NewInit = typename Reducer<Init, Head>::Type;
    using ReducedTail = typename TypeListReduce<TypeList<Tail...>, NewInit, Reducer>::Type;

public:
    using Type = ReducedTail;
};
} // namespace ATVC
#endif
