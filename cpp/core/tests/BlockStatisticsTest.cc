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
#include "shuffle/Payload.h"

#include <arrow/buffer.h>
#include <arrow/io/memory.h>
#include <arrow/type.h>
#include <arrow/util/bit_util.h>
#include <gtest/gtest.h>

#include <cstring>
#include <limits>
#include <vector>

namespace gluten {

class BlockStatisticsTest : public ::testing::Test {
 protected:
  // Build a validity buffer where bits at the given indices are unset (null).
  std::shared_ptr<arrow::Buffer> makeValidityBuffer(uint32_t numRows, const std::vector<uint32_t>& nullIndices) {
    auto byteCount = arrow::bit_util::BytesForBits(numRows);
    auto buf = arrow::AllocateBuffer(byteCount).ValueOrDie();
    // Start with all valid.
    memset(buf->mutable_data(), 0xFF, byteCount);
    for (auto idx : nullIndices) {
      arrow::bit_util::ClearBit(buf->mutable_data(), idx);
    }
    return buf;
  }

  // Build a value buffer from a vector of typed values.
  template <typename T>
  std::shared_ptr<arrow::Buffer> makeValueBuffer(const std::vector<T>& values) {
    auto byteSize = static_cast<int64_t>(values.size() * sizeof(T));
    auto buf = arrow::AllocateBuffer(byteSize).ValueOrDie();
    memcpy(buf->mutable_data(), values.data(), byteSize);
    return buf;
  }
};

TEST_F(BlockStatisticsTest, ColumnStatisticsSetGetInt64) {
  ColumnStatistics col{};
  col.setMin<int64_t>(-42);
  col.setMax<int64_t>(100);
  ASSERT_EQ(col.getMin<int64_t>(), -42);
  ASSERT_EQ(col.getMax<int64_t>(), 100);
}

TEST_F(BlockStatisticsTest, ColumnStatisticsSetGetFloat) {
  ColumnStatistics col{};
  col.setMin<float>(-1.5f);
  col.setMax<float>(3.14f);
  ASSERT_FLOAT_EQ(col.getMin<float>(), -1.5f);
  ASSERT_FLOAT_EQ(col.getMax<float>(), 3.14f);
}

TEST_F(BlockStatisticsTest, ColumnStatisticsSetGetDouble) {
  ColumnStatistics col{};
  col.setMin<double>(-99.99);
  col.setMax<double>(1e18);
  ASSERT_DOUBLE_EQ(col.getMin<double>(), -99.99);
  ASSERT_DOUBLE_EQ(col.getMax<double>(), 1e18);
}

TEST_F(BlockStatisticsTest, ColumnStatisticsSerializeDeserialize) {
  ColumnStatistics original{};
  original.columnIndex = 7;
  original.typeId = static_cast<uint8_t>(arrow::Type::INT64);
  original.hasNull = true;
  original.hasStats = true;
  original.setMin<int64_t>(-1000);
  original.setMax<int64_t>(2000);

  uint8_t buf[ColumnStatistics::kSerializedSize];
  uint8_t* ptr = buf;
  original.serialize(ptr);
  ASSERT_EQ(ptr - buf, ColumnStatistics::kSerializedSize);

  const uint8_t* readPtr = buf;
  auto restored = ColumnStatistics::deserialize(readPtr);
  ASSERT_EQ(readPtr - buf, ColumnStatistics::kSerializedSize);

  ASSERT_EQ(restored.columnIndex, 7);
  ASSERT_EQ(restored.typeId, static_cast<uint8_t>(arrow::Type::INT64));
  ASSERT_TRUE(restored.hasNull);
  ASSERT_TRUE(restored.hasStats);
  ASSERT_EQ(restored.getMin<int64_t>(), -1000);
  ASSERT_EQ(restored.getMax<int64_t>(), 2000);
}

TEST_F(BlockStatisticsTest, ColumnStatisticsMergeInt64) {
  ColumnStatistics a{};
  a.typeId = static_cast<uint8_t>(arrow::Type::INT64);
  a.hasNull = false;
  a.hasStats = true;
  a.setMin<int64_t>(10);
  a.setMax<int64_t>(50);

  ColumnStatistics b{};
  b.typeId = static_cast<uint8_t>(arrow::Type::INT64);
  b.hasNull = true;
  b.hasStats = true;
  b.setMin<int64_t>(5);
  b.setMax<int64_t>(30);

  a.merge(b);
  ASSERT_TRUE(a.hasNull);
  ASSERT_TRUE(a.hasStats);
  ASSERT_EQ(a.getMin<int64_t>(), 5);
  ASSERT_EQ(a.getMax<int64_t>(), 50);
}

TEST_F(BlockStatisticsTest, ColumnStatisticsMergeDouble) {
  ColumnStatistics a{};
  a.typeId = static_cast<uint8_t>(arrow::Type::DOUBLE);
  a.hasNull = false;
  a.hasStats = true;
  a.setMin<double>(1.5);
  a.setMax<double>(9.9);

  ColumnStatistics b{};
  b.typeId = static_cast<uint8_t>(arrow::Type::DOUBLE);
  b.hasNull = false;
  b.hasStats = true;
  b.setMin<double>(-0.5);
  b.setMax<double>(5.0);

  a.merge(b);
  ASSERT_FALSE(a.hasNull);
  ASSERT_DOUBLE_EQ(a.getMin<double>(), -0.5);
  ASSERT_DOUBLE_EQ(a.getMax<double>(), 9.9);
}

TEST_F(BlockStatisticsTest, ColumnStatisticsMergeOneEmpty) {
  ColumnStatistics a{};
  a.typeId = static_cast<uint8_t>(arrow::Type::INT32);
  a.hasNull = false;
  a.hasStats = false; // no stats yet

  ColumnStatistics b{};
  b.typeId = static_cast<uint8_t>(arrow::Type::INT32);
  b.hasNull = true;
  b.hasStats = true;
  b.setMin<int32_t>(100);
  b.setMax<int32_t>(200);

  a.merge(b);
  ASSERT_TRUE(a.hasNull);
  ASSERT_TRUE(a.hasStats);
  ASSERT_EQ(a.getMin<int32_t>(), 100);
  ASSERT_EQ(a.getMax<int32_t>(), 200);
}

TEST_F(BlockStatisticsTest, BlockStatisticsSerializeDeserialize) {
  BlockStatistics original;

  ColumnStatistics c0{};
  c0.columnIndex = 0;
  c0.typeId = static_cast<uint8_t>(arrow::Type::INT32);
  c0.hasNull = false;
  c0.hasStats = true;
  c0.setMin<int32_t>(-5);
  c0.setMax<int32_t>(42);

  ColumnStatistics c1{};
  c1.columnIndex = 1;
  c1.typeId = static_cast<uint8_t>(arrow::Type::DOUBLE);
  c1.hasNull = true;
  c1.hasStats = true;
  c1.setMin<double>(-1.0);
  c1.setMax<double>(99.5);

  original.columnStats.push_back(c0);
  original.columnStats.push_back(c1);

  const int64_t fakePayloadSize = 12345;

  // Serialize to an in-memory stream.
  auto sink = arrow::io::BufferOutputStream::Create().ValueOrDie();
  ASSERT_TRUE(original.serialize(sink.get(), fakePayloadSize).ok());
  auto serialized = sink->Finish().ValueOrDie();

  ASSERT_EQ(static_cast<uint32_t>(serialized->size()), original.serializedSize());

  // Deserialize.
  auto source = std::make_shared<arrow::io::BufferReader>(serialized);
  auto result = BlockStatistics::deserialize(source.get());
  ASSERT_TRUE(result.ok());

  auto& [restored, payloadSize] = result.ValueOrDie();
  ASSERT_EQ(payloadSize, fakePayloadSize);
  ASSERT_EQ(restored.columnStats.size(), 2u);

  ASSERT_EQ(restored.columnStats[0].columnIndex, 0);
  ASSERT_EQ(restored.columnStats[0].getMin<int32_t>(), -5);
  ASSERT_EQ(restored.columnStats[0].getMax<int32_t>(), 42);
  ASSERT_FALSE(restored.columnStats[0].hasNull);

  ASSERT_EQ(restored.columnStats[1].columnIndex, 1);
  ASSERT_DOUBLE_EQ(restored.columnStats[1].getMin<double>(), -1.0);
  ASSERT_DOUBLE_EQ(restored.columnStats[1].getMax<double>(), 99.5);
  ASSERT_TRUE(restored.columnStats[1].hasNull);
}

TEST_F(BlockStatisticsTest, BlockStatisticsSerializeEmpty) {
  BlockStatistics empty;

  auto sink = arrow::io::BufferOutputStream::Create().ValueOrDie();
  ASSERT_TRUE(empty.serialize(sink.get(), 0).ok());
  auto serialized = sink->Finish().ValueOrDie();

  auto source = std::make_shared<arrow::io::BufferReader>(serialized);
  auto result = BlockStatistics::deserialize(source.get());
  ASSERT_TRUE(result.ok());

  auto& [restored, payloadSize] = result.ValueOrDie();
  ASSERT_EQ(payloadSize, 0);
  ASSERT_TRUE(restored.columnStats.empty());
}

TEST_F(BlockStatisticsTest, BlockStatisticsMerge) {
  BlockStatistics a;
  {
    ColumnStatistics c{};
    c.columnIndex = 0;
    c.typeId = static_cast<uint8_t>(arrow::Type::INT64);
    c.hasNull = false;
    c.hasStats = true;
    c.setMin<int64_t>(10);
    c.setMax<int64_t>(20);
    a.columnStats.push_back(c);
  }

  BlockStatistics b;
  {
    ColumnStatistics c{};
    c.columnIndex = 0;
    c.typeId = static_cast<uint8_t>(arrow::Type::INT64);
    c.hasNull = true;
    c.hasStats = true;
    c.setMin<int64_t>(5);
    c.setMax<int64_t>(15);
    b.columnStats.push_back(c);
  }

  a.merge(b);
  ASSERT_EQ(a.columnStats.size(), 1u);
  ASSERT_TRUE(a.columnStats[0].hasNull);
  ASSERT_EQ(a.columnStats[0].getMin<int64_t>(), 5);
  ASSERT_EQ(a.columnStats[0].getMax<int64_t>(), 20);
}

TEST_F(BlockStatisticsTest, ComputeInt32Column) {
  // Schema: single INT32 column.
  auto schema = arrow::schema({arrow::field("id", arrow::int32())});

  uint32_t numRows = 5;
  std::vector<int32_t> values = {10, -3, 42, 7, 0};

  // Buffers: [validity(nullptr = all valid), value]
  std::vector<std::shared_ptr<arrow::Buffer>> buffers;
  buffers.push_back(nullptr); // all valid
  buffers.push_back(makeValueBuffer(values));

  auto stats = computeBlockStatistics(schema, buffers, numRows, /*hasComplexType=*/false);

  ASSERT_EQ(stats.columnStats.size(), 1u);
  auto& col = stats.columnStats[0];
  ASSERT_EQ(col.columnIndex, 0);
  ASSERT_FALSE(col.hasNull);
  ASSERT_TRUE(col.hasStats);
  ASSERT_EQ(col.getMin<int32_t>(), -3);
  ASSERT_EQ(col.getMax<int32_t>(), 42);
}

TEST_F(BlockStatisticsTest, ComputeInt64ColumnWithNulls) {
  auto schema = arrow::schema({arrow::field("id", arrow::int64())});

  uint32_t numRows = 4;
  std::vector<int64_t> values = {100, 200, 50, 300};

  // Row 2 is null.
  auto validity = makeValidityBuffer(numRows, {2});

  std::vector<std::shared_ptr<arrow::Buffer>> buffers;
  buffers.push_back(validity);
  buffers.push_back(makeValueBuffer(values));

  auto stats = computeBlockStatistics(schema, buffers, numRows, false);

  ASSERT_EQ(stats.columnStats.size(), 1u);
  auto& col = stats.columnStats[0];
  ASSERT_TRUE(col.hasNull);
  ASSERT_TRUE(col.hasStats);
  // Row 2 (value 50) is null and should be skipped.
  ASSERT_EQ(col.getMin<int64_t>(), 100);
  ASSERT_EQ(col.getMax<int64_t>(), 300);
}

TEST_F(BlockStatisticsTest, ComputeDoubleColumn) {
  auto schema = arrow::schema({arrow::field("val", arrow::float64())});

  uint32_t numRows = 3;
  std::vector<double> values = {-1.5, 0.0, 99.9};

  std::vector<std::shared_ptr<arrow::Buffer>> buffers;
  buffers.push_back(nullptr); // all valid
  buffers.push_back(makeValueBuffer(values));

  auto stats = computeBlockStatistics(schema, buffers, numRows, false);

  ASSERT_EQ(stats.columnStats.size(), 1u);
  ASSERT_TRUE(stats.columnStats[0].hasStats);
  ASSERT_DOUBLE_EQ(stats.columnStats[0].getMin<double>(), -1.5);
  ASSERT_DOUBLE_EQ(stats.columnStats[0].getMax<double>(), 99.9);
}

TEST_F(BlockStatisticsTest, ComputeFloatColumn) {
  auto schema = arrow::schema({arrow::field("val", arrow::float32())});

  uint32_t numRows = 4;
  std::vector<float> values = {2.5f, -0.1f, 7.0f, 3.0f};

  std::vector<std::shared_ptr<arrow::Buffer>> buffers;
  buffers.push_back(nullptr);
  buffers.push_back(makeValueBuffer(values));

  auto stats = computeBlockStatistics(schema, buffers, numRows, false);

  ASSERT_EQ(stats.columnStats.size(), 1u);
  ASSERT_TRUE(stats.columnStats[0].hasStats);
  ASSERT_FLOAT_EQ(stats.columnStats[0].getMin<float>(), -0.1f);
  ASSERT_FLOAT_EQ(stats.columnStats[0].getMax<float>(), 7.0f);
}

TEST_F(BlockStatisticsTest, ComputeMultipleColumns) {
  // Schema: INT32, DOUBLE
  auto schema = arrow::schema({arrow::field("a", arrow::int32()), arrow::field("b", arrow::float64())});

  uint32_t numRows = 3;
  std::vector<int32_t> ints = {5, -10, 20};
  std::vector<double> doubles = {1.0, 2.0, -3.0};

  // Buffer layout: [a_validity, a_value, b_validity, b_value]
  std::vector<std::shared_ptr<arrow::Buffer>> buffers;
  buffers.push_back(nullptr); // a validity
  buffers.push_back(makeValueBuffer(ints)); // a value
  buffers.push_back(nullptr); // b validity
  buffers.push_back(makeValueBuffer(doubles)); // b value

  auto stats = computeBlockStatistics(schema, buffers, numRows, false);

  ASSERT_EQ(stats.columnStats.size(), 2u);

  // Column a (INT32)
  ASSERT_EQ(stats.columnStats[0].columnIndex, 0);
  ASSERT_TRUE(stats.columnStats[0].hasStats);
  ASSERT_EQ(stats.columnStats[0].getMin<int32_t>(), -10);
  ASSERT_EQ(stats.columnStats[0].getMax<int32_t>(), 20);

  // Column b (DOUBLE)
  ASSERT_EQ(stats.columnStats[1].columnIndex, 1);
  ASSERT_TRUE(stats.columnStats[1].hasStats);
  ASSERT_DOUBLE_EQ(stats.columnStats[1].getMin<double>(), -3.0);
  ASSERT_DOUBLE_EQ(stats.columnStats[1].getMax<double>(), 2.0);
}

TEST_F(BlockStatisticsTest, ComputeWithStringColumn) {
  // Schema: INT32, STRING
  // Strings produce 3 buffers (validity, length, value) but no min/max stats.
  auto schema = arrow::schema({arrow::field("id", arrow::int32()), arrow::field("name", arrow::utf8())});

  uint32_t numRows = 2;
  std::vector<int32_t> ints = {1, 2};

  // Buffer layout: [id_validity, id_value, name_validity, name_length, name_value]
  std::vector<std::shared_ptr<arrow::Buffer>> buffers;
  buffers.push_back(nullptr); // id validity
  buffers.push_back(makeValueBuffer(ints)); // id value
  buffers.push_back(nullptr); // name validity
  buffers.push_back(arrow::AllocateBuffer(numRows * sizeof(uint32_t)).ValueOrDie()); // name length
  buffers.push_back(arrow::AllocateBuffer(0).ValueOrDie()); // name value (empty)

  auto stats = computeBlockStatistics(schema, buffers, numRows, false);

  ASSERT_EQ(stats.columnStats.size(), 2u);

  // Column 0 (INT32) — has stats.
  ASSERT_TRUE(stats.columnStats[0].hasStats);
  ASSERT_EQ(stats.columnStats[0].getMin<int32_t>(), 1);
  ASSERT_EQ(stats.columnStats[0].getMax<int32_t>(), 2);

  // Column 1 (STRING) — no min/max stats, but tracks nullability.
  ASSERT_FALSE(stats.columnStats[1].hasStats);
  ASSERT_FALSE(stats.columnStats[1].hasNull);
}

TEST_F(BlockStatisticsTest, ComputeAllNullColumn) {
  auto schema = arrow::schema({arrow::field("x", arrow::int64())});

  uint32_t numRows = 3;
  std::vector<int64_t> values = {0, 0, 0}; // values don't matter, all null
  auto validity = makeValidityBuffer(numRows, {0, 1, 2}); // all null

  std::vector<std::shared_ptr<arrow::Buffer>> buffers;
  buffers.push_back(validity);
  buffers.push_back(makeValueBuffer(values));

  auto stats = computeBlockStatistics(schema, buffers, numRows, false);

  ASSERT_EQ(stats.columnStats.size(), 1u);
  ASSERT_TRUE(stats.columnStats[0].hasNull);
  ASSERT_FALSE(stats.columnStats[0].hasStats); // No non-null values → no min/max.
}

TEST_F(BlockStatisticsTest, ComputeEmptyBlock) {
  auto schema = arrow::schema({arrow::field("x", arrow::int32())});

  std::vector<std::shared_ptr<arrow::Buffer>> buffers;

  auto stats = computeBlockStatistics(schema, buffers, /*numRows=*/0, false);
  ASSERT_TRUE(stats.columnStats.empty());
}

TEST_F(BlockStatisticsTest, ComputeSingleRow) {
  auto schema = arrow::schema({arrow::field("x", arrow::int32())});

  uint32_t numRows = 1;
  std::vector<int32_t> values = {77};

  std::vector<std::shared_ptr<arrow::Buffer>> buffers;
  buffers.push_back(nullptr);
  buffers.push_back(makeValueBuffer(values));

  auto stats = computeBlockStatistics(schema, buffers, numRows, false);

  ASSERT_EQ(stats.columnStats.size(), 1u);
  ASSERT_TRUE(stats.columnStats[0].hasStats);
  ASSERT_EQ(stats.columnStats[0].getMin<int32_t>(), 77);
  ASSERT_EQ(stats.columnStats[0].getMax<int32_t>(), 77);
}

TEST_F(BlockStatisticsTest, ComputeNegativeValues) {
  auto schema = arrow::schema({arrow::field("x", arrow::int32())});

  uint32_t numRows = 4;
  std::vector<int32_t> values = {-100, -50, -200, -1};

  std::vector<std::shared_ptr<arrow::Buffer>> buffers;
  buffers.push_back(nullptr);
  buffers.push_back(makeValueBuffer(values));

  auto stats = computeBlockStatistics(schema, buffers, numRows, false);

  ASSERT_EQ(stats.columnStats[0].getMin<int32_t>(), -200);
  ASSERT_EQ(stats.columnStats[0].getMax<int32_t>(), -1);
}

TEST_F(BlockStatisticsTest, ComputeInt8Column) {
  auto schema = arrow::schema({arrow::field("x", arrow::int8())});

  uint32_t numRows = 3;
  std::vector<int8_t> values = {-128, 0, 127};

  std::vector<std::shared_ptr<arrow::Buffer>> buffers;
  buffers.push_back(nullptr);
  buffers.push_back(makeValueBuffer(values));

  auto stats = computeBlockStatistics(schema, buffers, numRows, false);

  ASSERT_EQ(stats.columnStats.size(), 1u);
  ASSERT_TRUE(stats.columnStats[0].hasStats);
  ASSERT_EQ(stats.columnStats[0].getMin<int8_t>(), -128);
  ASSERT_EQ(stats.columnStats[0].getMax<int8_t>(), 127);
}

TEST_F(BlockStatisticsTest, ComputeInt16Column) {
  auto schema = arrow::schema({arrow::field("x", arrow::int16())});

  uint32_t numRows = 3;
  std::vector<int16_t> values = {-1000, 500, 32000};

  std::vector<std::shared_ptr<arrow::Buffer>> buffers;
  buffers.push_back(nullptr);
  buffers.push_back(makeValueBuffer(values));

  auto stats = computeBlockStatistics(schema, buffers, numRows, false);

  ASSERT_EQ(stats.columnStats[0].getMin<int16_t>(), -1000);
  ASSERT_EQ(stats.columnStats[0].getMax<int16_t>(), 32000);
}

TEST_F(BlockStatisticsTest, InMemoryPayloadCarriesStats) {
  auto schema = arrow::schema({arrow::field("x", arrow::int32())});

  uint32_t numRows = 3;
  std::vector<int32_t> values = {1, 2, 3};

  std::vector<std::shared_ptr<arrow::Buffer>> buffers;
  buffers.push_back(nullptr);
  buffers.push_back(makeValueBuffer(values));

  std::vector<bool> isValidityBuffer = {true, false};

  auto payload = std::make_unique<InMemoryPayload>(
      numRows, &isValidityBuffer, schema, std::move(buffers), /*hasComplexType=*/false);

  ASSERT_FALSE(payload->hasBlockStats());

  // Compute and set stats.
  auto stats = computeBlockStatistics(schema, payload->getBuffers(), numRows, false);
  payload->setBlockStats(std::move(stats));

  ASSERT_TRUE(payload->hasBlockStats());
  ASSERT_EQ(payload->blockStats()->columnStats.size(), 1u);
  ASSERT_EQ(payload->blockStats()->columnStats[0].getMin<int32_t>(), 1);
  ASSERT_EQ(payload->blockStats()->columnStats[0].getMax<int32_t>(), 3);

  // Convert to BlockPayload — stats should survive.
  auto pool = arrow::default_memory_pool();
  auto blockResult = payload->toBlockPayload(Payload::kUncompressed, pool, nullptr);
  ASSERT_TRUE(blockResult.ok());
  auto blockPayload = std::move(blockResult).ValueOrDie();

  ASSERT_TRUE(blockPayload->hasBlockStats());
  ASSERT_EQ(blockPayload->blockStats()->columnStats[0].getMin<int32_t>(), 1);
  ASSERT_EQ(blockPayload->blockStats()->columnStats[0].getMax<int32_t>(), 3);
}

TEST_F(BlockStatisticsTest, InMemoryPayloadMergePreservesStats) {
  auto schema = arrow::schema({arrow::field("x", arrow::int64())});
  std::vector<bool> isValidityBuffer = {true, false};
  auto pool = arrow::default_memory_pool();

  // Payload A: values [10, 20]
  {
    uint32_t numRows = 2;
    std::vector<int64_t> values = {10, 20};
    std::vector<std::shared_ptr<arrow::Buffer>> buffers;
    buffers.push_back(nullptr);
    buffers.push_back(makeValueBuffer(values));
    auto a = std::make_unique<InMemoryPayload>(numRows, &isValidityBuffer, schema, std::move(buffers));
    auto statsA = computeBlockStatistics(schema, a->getBuffers(), numRows, false);
    a->setBlockStats(std::move(statsA));

    // Payload B: values [5, 15]
    std::vector<int64_t> valuesB = {5, 15};
    std::vector<std::shared_ptr<arrow::Buffer>> buffersB;
    buffersB.push_back(nullptr);
    buffersB.push_back(makeValueBuffer(valuesB));
    auto b = std::make_unique<InMemoryPayload>(numRows, &isValidityBuffer, schema, std::move(buffersB));
    auto statsB = computeBlockStatistics(schema, b->getBuffers(), numRows, false);
    b->setBlockStats(std::move(statsB));

    // Merge.
    auto merged = InMemoryPayload::merge(std::move(a), std::move(b), pool);
    ASSERT_TRUE(merged.ok());
    auto mergedPayload = std::move(merged).ValueOrDie();

    ASSERT_TRUE(mergedPayload->hasBlockStats());
    ASSERT_EQ(mergedPayload->blockStats()->columnStats.size(), 1u);
    ASSERT_EQ(mergedPayload->blockStats()->columnStats[0].getMin<int64_t>(), 5);
    ASSERT_EQ(mergedPayload->blockStats()->columnStats[0].getMax<int64_t>(), 20);
  }
}

TEST_F(BlockStatisticsTest, BlockPayloadSerializedSize) {
  auto pool = arrow::default_memory_pool();
  std::vector<bool> isValidityBuffer = {true, false};

  uint32_t numRows = 2;
  std::vector<int32_t> values = {1, 2};

  std::vector<std::shared_ptr<arrow::Buffer>> buffers;
  buffers.push_back(nullptr); // validity
  buffers.push_back(makeValueBuffer(values)); // value

  auto result = BlockPayload::fromBuffers(Payload::kUncompressed, numRows, std::move(buffers), &isValidityBuffer, pool, nullptr);
  ASSERT_TRUE(result.ok());
  auto payload = std::move(result).ValueOrDie();

  int64_t expectedSize = payload->serializedSize();
  ASSERT_GT(expectedSize, 0);

  // Serialize and verify the actual size matches.
  auto sink = arrow::io::BufferOutputStream::Create().ValueOrDie();
  ASSERT_TRUE(payload->serialize(sink.get()).ok());
  auto written = sink->Finish().ValueOrDie();

  ASSERT_EQ(written->size(), expectedSize);
}

} // namespace gluten
