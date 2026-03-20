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

#include "shuffle/BlockStatistics.h"

#include <arrow/buffer.h>
#include <arrow/type.h>
#include <arrow/util/bit_util.h>

namespace gluten {
namespace {

// Returns true if the row at the given index is valid (non-null).
inline bool isRowValid(const std::shared_ptr<arrow::Buffer>& validityBuffer, uint32_t row) {
  if (!validityBuffer) {
    return true; // No validity buffer means all rows are valid.
  }
  return arrow::bit_util::GetBit(validityBuffer->data(), row);
}

// Returns true if the column has any null rows.
bool hasAnyNull(const std::shared_ptr<arrow::Buffer>& validityBuffer, uint32_t numRows) {
  if (!validityBuffer || numRows == 0) {
    return false;
  }
  const uint8_t* data = validityBuffer->data();
  uint32_t fullBytes = numRows / 8;
  // Check full bytes — 0xFF means all 8 bits are valid.
  for (uint32_t i = 0; i < fullBytes; ++i) {
    if (data[i] != 0xFF) {
      return true;
    }
  }
  // Check remaining bits in the last partial byte.
  uint32_t remainingBits = numRows % 8;
  if (remainingBits > 0) {
    uint8_t mask = static_cast<uint8_t>((1u << remainingBits) - 1);
    if ((data[fullBytes] & mask) != mask) {
      return true;
    }
  }
  return false;
}

template <typename T>
void writeBytes(uint8_t*& dst, T value) {
  memcpy(dst, &value, sizeof(T));
  dst += sizeof(T);
}

template <typename T>
T readBytes(const uint8_t*& src) {
  T value;
  memcpy(&value, src, sizeof(T));
  src += sizeof(T);
  return value;
}

template <typename T>
void scanColumnMinMax(
    const std::shared_ptr<arrow::Buffer>& validityBuffer,
    const std::shared_ptr<arrow::Buffer>& valueBuffer,
    uint32_t numRows,
    ColumnStatistics& stats) {
  if (!valueBuffer || valueBuffer->size() == 0 || numRows == 0) {
    return;
  }

  const auto* values = reinterpret_cast<const T*>(valueBuffer->data());
  bool foundAny = false;
  T minVal{};
  T maxVal{};

  for (uint32_t i = 0; i < numRows; ++i) {
    if (!isRowValid(validityBuffer, i)) {
      continue;
    }
    T val = values[i];
    if (!foundAny) {
      minVal = val;
      maxVal = val;
      foundAny = true;
    } else {
      if (val < minVal) {
        minVal = val;
      }
      if (val > maxVal) {
        maxVal = val;
      }
    }
  }

  if (foundAny) {
    stats.hasStats = true;
    stats.setMin(minVal);
    stats.setMax(maxVal);
  }
}

} // namespace

void ColumnStatistics::merge(const ColumnStatistics& other) {
  hasNull = hasNull || other.hasNull;
  if (!other.hasStats) {
    return;
  }
  if (!hasStats) {
    hasStats = true;
    memcpy(minBytes, other.minBytes, 8);
    memcpy(maxBytes, other.maxBytes, 8);
    return;
  }
  // Both have stats — merge based on type.
  switch (static_cast<arrow::Type::type>(typeId)) {
    case arrow::Type::INT8:
      mergeTyped<int8_t>(other);
      break;
    case arrow::Type::INT16:
      mergeTyped<int16_t>(other);
      break;
    case arrow::Type::INT32:
    case arrow::Type::DATE32:
      mergeTyped<int32_t>(other);
      break;
    case arrow::Type::INT64:
    case arrow::Type::DATE64:
    case arrow::Type::TIMESTAMP:
      mergeTyped<int64_t>(other);
      break;
    case arrow::Type::FLOAT:
      mergeTyped<float>(other);
      break;
    case arrow::Type::DOUBLE:
      mergeTyped<double>(other);
      break;
    default:
      break;
  }
}

arrow::Status BlockStatistics::serialize(arrow::io::OutputStream* out, int64_t payloadSize) const {
  uint32_t size = serializedSize();
  std::vector<uint8_t> buffer(size);
  uint8_t* ptr = buffer.data();

  writeBytes(ptr, kVersion);
  writeBytes(ptr, static_cast<uint16_t>(columnStats.size()));
  writeBytes(ptr, payloadSize);

  for (const auto& col : columnStats) {
    col.serialize(ptr);
  }

  return out->Write(buffer.data(), size);
}

arrow::Result<std::pair<BlockStatistics, int64_t>> BlockStatistics::deserialize(arrow::io::InputStream* in) {
  // Read version.
  uint8_t version;
  ARROW_ASSIGN_OR_RAISE(auto bytesRead, in->Read(sizeof(version), &version));
  if (bytesRead != sizeof(version) || version != kVersion) {
    return arrow::Status::Invalid("Unsupported BlockStatistics version: ", static_cast<int>(version));
  }

  // Read numColumns.
  uint16_t numColumns;
  ARROW_ASSIGN_OR_RAISE(bytesRead, in->Read(sizeof(numColumns), &numColumns));
  if (bytesRead != sizeof(numColumns)) {
    return arrow::Status::IOError("Unexpected end of stream reading BlockStatistics numColumns");
  }

  // Read payloadSize.
  int64_t payloadSize;
  ARROW_ASSIGN_OR_RAISE(bytesRead, in->Read(sizeof(payloadSize), &payloadSize));
  if (bytesRead != sizeof(payloadSize)) {
    return arrow::Status::IOError("Unexpected end of stream reading BlockStatistics payloadSize");
  }

  BlockStatistics stats;
  stats.columnStats.reserve(numColumns);

  for (uint16_t i = 0; i < numColumns; ++i) {
    uint8_t buf[ColumnStatistics::kSerializedSize];
    ARROW_ASSIGN_OR_RAISE(bytesRead, in->Read(sizeof(buf), buf));
    if (bytesRead != sizeof(buf)) {
      return arrow::Status::IOError("Unexpected end of stream reading BlockStatistics column ", i);
    }
    const uint8_t* ptr = buf;
    stats.columnStats.push_back(ColumnStatistics::deserialize(ptr));
  }

  return std::make_pair(std::move(stats), payloadSize);
}

void BlockStatistics::merge(const BlockStatistics& other) {
  for (size_t i = 0; i < columnStats.size() && i < other.columnStats.size(); ++i) {
    columnStats[i].merge(other.columnStats[i]);
  }
}

BlockStatistics computeBlockStatistics(
    const std::shared_ptr<arrow::Schema>& schema,
    const std::vector<std::shared_ptr<arrow::Buffer>>& buffers,
    uint32_t numRows,
    bool hasComplexType) {
  BlockStatistics result;
  if (numRows == 0 || buffers.empty()) {
    return result;
  }

  uint32_t bufIdx = 0;
  auto numFields = schema->num_fields();

  for (int fieldIdx = 0; fieldIdx < numFields; ++fieldIdx) {
    auto typeId = schema->field(fieldIdx)->type()->id();

    switch (typeId) {
      case arrow::Type::BINARY:
      case arrow::Type::STRING:
      case arrow::Type::LARGE_BINARY:
      case arrow::Type::LARGE_STRING: {
        if (bufIdx + 3 > buffers.size()) {
          break;
        }
        auto validityBuf = buffers[bufIdx++]; // validity
        bufIdx++; // length (skip)
        bufIdx++; // value (skip)

        ColumnStatistics col{};
        col.columnIndex = static_cast<uint16_t>(fieldIdx);
        col.typeId = static_cast<uint8_t>(typeId);
        col.hasNull = hasAnyNull(validityBuf, numRows);
        col.hasStats = false; // String stats not supported yet.
        result.columnStats.push_back(col);
        break;
      }
      case arrow::Type::STRUCT:
      case arrow::Type::MAP:
      case arrow::Type::LIST:
      case arrow::Type::LARGE_LIST:
        // Complex types are skipped in assembleBuffers() per-field loop.
        // Their buffer is appended at the end. No stats for them.
        break;
      case arrow::Type::NA:
        // Null type has no buffers.
        break;
      case arrow::Type::BOOL: {
        if (bufIdx + 2 > buffers.size()) {
          break;
        }
        auto validityBuf = buffers[bufIdx++]; // validity
        bufIdx++; // value (bit-packed, skip for stats)

        ColumnStatistics col{};
        col.columnIndex = static_cast<uint16_t>(fieldIdx);
        col.typeId = static_cast<uint8_t>(typeId);
        col.hasNull = hasAnyNull(validityBuf, numRows);
        col.hasStats = false; // Bool stats not useful.
        result.columnStats.push_back(col);
        break;
      }
      default: {
        // Fixed-width numeric types.
        if (bufIdx + 2 > buffers.size()) {
          break;
        }
        auto validityBuf = buffers[bufIdx++]; // validity
        auto valueBuf = buffers[bufIdx++]; // value

        ColumnStatistics col{};
        col.columnIndex = static_cast<uint16_t>(fieldIdx);
        col.typeId = static_cast<uint8_t>(typeId);
        col.hasNull = hasAnyNull(validityBuf, numRows);
        col.hasStats = false;

        switch (typeId) {
          case arrow::Type::INT8:
            scanColumnMinMax<int8_t>(validityBuf, valueBuf, numRows, col);
            break;
          case arrow::Type::INT16:
            scanColumnMinMax<int16_t>(validityBuf, valueBuf, numRows, col);
            break;
          case arrow::Type::INT32:
          case arrow::Type::DATE32:
            scanColumnMinMax<int32_t>(validityBuf, valueBuf, numRows, col);
            break;
          case arrow::Type::INT64:
          case arrow::Type::DATE64:
          case arrow::Type::TIMESTAMP:
            scanColumnMinMax<int64_t>(validityBuf, valueBuf, numRows, col);
            break;
          case arrow::Type::FLOAT:
            scanColumnMinMax<float>(validityBuf, valueBuf, numRows, col);
            break;
          case arrow::Type::DOUBLE:
            scanColumnMinMax<double>(validityBuf, valueBuf, numRows, col);
            break;
          default:
            // Unsupported type for min/max stats.
            break;
        }

        result.columnStats.push_back(col);
        break;
      }
    }
  }

  return result;
}

} // namespace gluten
