/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <arrow/io/interfaces.h>
#include <arrow/result.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

namespace arrow {
class Buffer;
class Schema;
} // namespace arrow

namespace gluten {

/// Per-column min/max statistics for a shuffle block.
struct ColumnStatistics {
  uint16_t columnIndex; // Index in the schema.
  uint8_t typeId; // Arrow type ID.
  bool hasNull; // Whether the column contains nulls in this block.
  bool hasStats; // Whether min/max are valid (false for unsupported types).

  // Raw bytes for min/max. Interpretation depends on typeId.
  uint8_t minBytes[8]{};
  uint8_t maxBytes[8]{};

  static constexpr uint32_t kSerializedSize = sizeof(uint16_t) + // columnIndex
      sizeof(uint8_t) + // typeId
      sizeof(uint8_t) + // flags (hasNull | hasStats)
      8 + // min
      8; // max

  template <typename T>
  void setMin(T value) {
    static_assert(sizeof(T) <= 8);
    memset(minBytes, 0, 8);
    memcpy(minBytes, &value, sizeof(T));
  }

  template <typename T>
  void setMax(T value) {
    static_assert(sizeof(T) <= 8);
    memset(maxBytes, 0, 8);
    memcpy(maxBytes, &value, sizeof(T));
  }

  template <typename T>
  T getMin() const {
    T value{};
    memcpy(&value, minBytes, sizeof(T));
    return value;
  }

  template <typename T>
  T getMax() const {
    T value{};
    memcpy(&value, maxBytes, sizeof(T));
    return value;
  }

  void serialize(uint8_t*& dst) const {
    memcpy(dst, &columnIndex, sizeof(columnIndex));
    dst += sizeof(columnIndex);
    memcpy(dst, &typeId, sizeof(typeId));
    dst += sizeof(typeId);
    uint8_t flags = (hasNull ? 1u : 0u) | (hasStats ? 2u : 0u);
    memcpy(dst, &flags, sizeof(flags));
    dst += sizeof(flags);
    memcpy(dst, minBytes, 8);
    dst += 8;
    memcpy(dst, maxBytes, 8);
    dst += 8;
  }

  static ColumnStatistics deserialize(const uint8_t*& src) {
    ColumnStatistics stats{};
    memcpy(&stats.columnIndex, src, sizeof(stats.columnIndex));
    src += sizeof(stats.columnIndex);
    memcpy(&stats.typeId, src, sizeof(stats.typeId));
    src += sizeof(stats.typeId);
    uint8_t flags;
    memcpy(&flags, src, sizeof(flags));
    src += sizeof(flags);
    stats.hasNull = (flags & 1u) != 0;
    stats.hasStats = (flags & 2u) != 0;
    memcpy(stats.minBytes, src, 8);
    src += 8;
    memcpy(stats.maxBytes, src, 8);
    src += 8;
    return stats;
  }

  /// Merge another ColumnStatistics into this one (for merging payloads).
  void merge(const ColumnStatistics& other);

 private:
  template <typename T>
  void mergeTyped(const ColumnStatistics& other) {
    auto myMin = getMin<T>();
    auto otherMin = other.getMin<T>();
    if (otherMin < myMin) {
      setMin(otherMin);
    }
    auto myMax = getMax<T>();
    auto otherMax = other.getMax<T>();
    if (otherMax > myMax) {
      setMax(otherMax);
    }
  }
};

/// Block-level statistics containing per-column min/max for a shuffle block.
struct BlockStatistics {
  static constexpr uint8_t kVersion = 1;

  std::vector<ColumnStatistics> columnStats;

  /// Byte size of the serialized stats header (excluding the BlockType byte).
  uint32_t serializedSize() const {
    return sizeof(uint8_t) + // version
        sizeof(uint16_t) + // numColumns
        sizeof(int64_t) + // payloadSize
        static_cast<uint32_t>(columnStats.size()) * ColumnStatistics::kSerializedSize;
  }

  /// Serialize to output stream. payloadSize is the byte size of the
  /// following payload block (BlockType byte + serialized payload data).
  arrow::Status serialize(arrow::io::OutputStream* out, int64_t payloadSize) const;

  /// Deserialize from input stream. Returns (stats, payloadSize).
  static arrow::Result<std::pair<BlockStatistics, int64_t>> deserialize(arrow::io::InputStream* in);

  /// Merge another BlockStatistics into this one.
  void merge(const BlockStatistics& other);
};

/// Compute block-level statistics from assembled Arrow buffers.
/// The buffer layout must match the assembleBuffers() output:
/// for each field in schema order, fixed-width fields produce
/// [validity, value], binary fields produce [validity, length, value],
/// null/complex types are skipped (complex buffer appended at end).
BlockStatistics computeBlockStatistics(
    const std::shared_ptr<arrow::Schema>& schema,
    const std::vector<std::shared_ptr<arrow::Buffer>>& buffers,
    uint32_t numRows,
    bool hasComplexType);

} // namespace gluten
