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
package org.apache.gluten.execution.tpch

import org.apache.gluten.backendsapi.clickhouse.CHBackendSettings
import org.apache.gluten.execution._

import org.apache.spark.SparkConf
import org.apache.spark.sql.catalyst.optimizer.BuildLeft
import org.apache.spark.sql.catalyst.plans.LeftSemi
import org.apache.spark.sql.execution.{ReusedSubqueryExec, SubqueryExec}
import org.apache.spark.sql.execution.adaptive.AdaptiveSparkPlanExec

class GlutenClickHouseTPCHParquetAQESuite extends ParquetTPCHSuite {

  /** Run Gluten + ClickHouse Backend with SortShuffleManager */
  override protected def sparkConf: SparkConf = {
    import org.apache.gluten.backendsapi.clickhouse.CHConfig._

    super.sparkConf
      .set("spark.shuffle.manager", "sort")
      .set("spark.io.compression.codec", "snappy")
      .set("spark.sql.shuffle.partitions", "5")
      .set("spark.sql.autoBroadcastJoinThreshold", "10MB")
      .set("spark.sql.adaptive.enabled", "true")
      .setCHConfig("use_local_format", true)
      .set(CHBackendSettings.GLUTEN_CLICKHOUSE_SHUFFLE_HASH_ALGORITHM, "sparkMurmurHash3_32")
  }

  final override val testCases: Seq[Int] = Seq(
    4, 6, 10, 12, 13, 16, 19, 20, 21
  )
  final override val testCasesWithConfig: Map[Int, Seq[(String, String)]] =
    Map(
      7 -> Seq(
        ("spark.sql.shuffle.partitions", "2"),
        ("spark.sql.autoBroadcastJoinThreshold", "-1")),
      8 -> Seq(
        ("spark.sql.shuffle.partitions", "1"),
        ("spark.sql.autoBroadcastJoinThreshold", "-1")),
      14 -> Seq(
        ("spark.sql.shuffle.partitions", "1"),
        ("spark.sql.autoBroadcastJoinThreshold", "-1")),
      17 -> Seq(("spark.shuffle.sort.bypassMergeThreshold", "2")),
      18 -> Seq(("spark.shuffle.sort.bypassMergeThreshold", "2"))
    )
  setupTestCase()

  test("TPCH Q1") {
    customCheck(1) {
      df =>
        assert(df.queryExecution.executedPlan.isInstanceOf[AdaptiveSparkPlanExec])
        val scanExec = collect(df.queryExecution.executedPlan) {
          case scanExec: BasicScanExecTransformer => scanExec
        }
        assert(scanExec.size == 1)
    }
  }

  test("TPCH Q2") {
    customCheck(2) {
      df =>
        assert(df.queryExecution.executedPlan.isInstanceOf[AdaptiveSparkPlanExec])
        val scanExec = collect(df.queryExecution.executedPlan) {
          case scanExec: BasicScanExecTransformer => scanExec
        }
        assert(scanExec.size == 8)
    }
  }

  test("TPCH Q3") {
    withSQLConf(("spark.sql.autoBroadcastJoinThreshold", "-1")) {
      customCheck(3) {
        df =>
          assert(df.queryExecution.executedPlan.isInstanceOf[AdaptiveSparkPlanExec])
          val shjBuildLeft = collect(df.queryExecution.executedPlan) {
            case shj: ShuffledHashJoinExecTransformerBase if shj.joinBuildSide == BuildLeft => shj
          }
          assert(shjBuildLeft.size == 2)
      }
    }
  }

  test("TPCH Q5") {
    withSQLConf(("spark.sql.autoBroadcastJoinThreshold", "-1")) {
      customCheck(5) {
        df =>
          assert(df.queryExecution.executedPlan.isInstanceOf[AdaptiveSparkPlanExec])
          val bhjRes = collect(df.queryExecution.executedPlan) {
            case bhj: BroadcastHashJoinExecTransformerBase => bhj
          }
          assert(bhjRes.isEmpty)
      }
    }
  }

  /**
   * TODO: With Spark 3.3, it can not support to use Spark Shuffle Manager and set
   * shuffle.partitions=1 at the same time, because OptimizeOneRowPlan rule will remove Sort
   * operator.
   */
  ignore("TPCH Q7 - with shuffle.partitions=1") {
    withSQLConf(
      ("spark.sql.shuffle.partitions", "1"),
      ("spark.sql.autoBroadcastJoinThreshold", "-1")) {
      check(7)
    }
  }

  test("TPCH Q9") {
    check(9, compare = false)
  }

  test("TPCH Q11") {
    customCheck(11, compare = false) {
      df =>
        assert(df.queryExecution.executedPlan.isInstanceOf[AdaptiveSparkPlanExec])
        val adaptiveSparkPlanExec = collectWithSubqueries(df.queryExecution.executedPlan) {
          case adaptive: AdaptiveSparkPlanExec => adaptive
        }
        assert(adaptiveSparkPlanExec.size == 2)
    }
  }

  test("TPCH Q15") {
    customCheck(15) {
      df =>
        assert(df.queryExecution.executedPlan.isInstanceOf[AdaptiveSparkPlanExec])
        val adaptiveSparkPlanExec = collectWithSubqueries(df.queryExecution.executedPlan) {
          case adaptive: AdaptiveSparkPlanExec => adaptive
        }
        assert(adaptiveSparkPlanExec.size == 2)
    }
  }

  test("GLUTEN-7971:Q21 Support using left side as the build table for the left anti/semi join") {
    withSQLConf(
      ("spark.sql.autoBroadcastJoinThreshold", "-1"),
      (CHBackendSettings.GLUTEN_CLICKHOUSE_CONVERT_LEFT_ANTI_SEMI_TO_RIGHT, "true")) {
      customCheck(21, compare = false) {
        df =>
          assert(df.queryExecution.executedPlan.isInstanceOf[AdaptiveSparkPlanExec])
          val shuffledHashJoinExecs = collect(df.queryExecution.executedPlan) {
            case h: CHShuffledHashJoinExecTransformer if h.joinType == LeftSemi => h
          }
          assertResult(1)(shuffledHashJoinExecs.size)
          assertResult(BuildLeft)(shuffledHashJoinExecs.head.buildSide)
      }
    }
  }

  test("TPCH Q22") {
    customCheck(22) {
      df =>
        assert(df.queryExecution.executedPlan.isInstanceOf[AdaptiveSparkPlanExec])
        val adaptiveSparkPlanExec = collectWithSubqueries(df.queryExecution.executedPlan) {
          case adaptive: AdaptiveSparkPlanExec => adaptive
        }
        assert(adaptiveSparkPlanExec.size == 2)
    }
  }

  test("GLUTEN-1620: fix 'attribute binding failed.' when executing hash agg with aqe") {
    val sql =
      """
        |SELECT *
        |    FROM (
        |        SELECT t1.O_ORDERSTATUS,
        |               t4.ACTIVECUSTOMERS / t1.ACTIVECUSTOMERS AS REPEATPURCHASERATE
        |        FROM (
        |            SELECT o_orderstatus AS O_ORDERSTATUS, COUNT(1) AS ACTIVECUSTOMERS
        |            FROM orders
        |            GROUP BY o_orderstatus
        |        ) t1
        |            INNER JOIN (
        |                SELECT o_orderstatus AS O_ORDERSTATUS, MAX(o_totalprice) AS ACTIVECUSTOMERS
        |                FROM orders
        |                GROUP BY o_orderstatus
        |            ) t4
        |            ON t1.O_ORDERSTATUS = t4.O_ORDERSTATUS
        |   ) t5
        |        INNER JOIN (
        |            SELECT t8.O_ORDERSTATUS,
        |                   t9.ACTIVECUSTOMERS / t8.ACTIVECUSTOMERS AS REPEATPURCHASERATE
        |            FROM (
        |                SELECT o_orderstatus AS O_ORDERSTATUS, COUNT(1) AS ACTIVECUSTOMERS
        |                FROM orders
        |                GROUP BY o_orderstatus
        |            ) t8
        |                INNER JOIN (
        |                    SELECT o_orderstatus AS O_ORDERSTATUS,
        |                          MAX(o_totalprice) AS ACTIVECUSTOMERS
        |                    FROM orders
        |                    GROUP BY o_orderstatus
        |                ) t9
        |                ON t8.O_ORDERSTATUS = t9.O_ORDERSTATUS
        |            ) t12
        |        ON t5.O_ORDERSTATUS = t12.O_ORDERSTATUS
        |""".stripMargin
    compareResultsAgainstVanillaSpark(sql, true, { df => })
  }

  test("GLUTEN-1848: Fix execute subquery repeatedly issue with ReusedSubquery") {
    val sql =
      """
        |SELECT
        |    s_suppkey,
        |    s_name,
        |    s_address,
        |    s_phone,
        |    total_revenue,
        |    total_revenue_1
        |FROM
        |    supplier,
        |    (
        |        SELECT
        |            l_suppkey AS supplier_no,
        |            sum(l_extendedprice * (1 - l_discount)) AS total_revenue,
        |            sum(l_extendedprice * l_discount) AS total_revenue_1
        |        FROM
        |            lineitem
        |        WHERE
        |            l_shipdate >= date'1996-01-01' AND
        |            l_shipdate < date'1996-01-01' + interval 3 month
        |        GROUP BY
        |            supplier_no) revenue0
        |WHERE
        |    s_suppkey = supplier_no
        |    AND total_revenue = (
        |        SELECT
        |            max(total_revenue)
        |        FROM (
        |            SELECT
        |                l_suppkey AS supplier_no,
        |                sum(l_extendedprice * (1 - l_discount)) AS total_revenue
        |            FROM
        |                lineitem
        |            WHERE
        |                l_shipdate >= date'1996-01-01' AND
        |                l_shipdate < date'1996-01-01' + interval 3 month
        |            GROUP BY
        |                supplier_no) revenue1)
        |    AND total_revenue_1 < (
        |        SELECT
        |            max(total_revenue)
        |        FROM (
        |            SELECT
        |                l_suppkey AS supplier_no,
        |                sum(l_extendedprice * (1 - l_discount)) AS total_revenue
        |            FROM
        |                lineitem
        |            WHERE
        |                l_shipdate >= date'1996-01-01' AND
        |                l_shipdate < date'1996-01-01' + interval 3 month
        |            GROUP BY
        |                supplier_no) revenue1)
        |ORDER BY
        |    s_suppkey;
        |""".stripMargin
    compareResultsAgainstVanillaSpark(
      sql,
      true,
      {
        df =>
          val subqueriesId = collectWithSubqueries(df.queryExecution.executedPlan) {
            case s: SubqueryExec => s.id
            case rs: ReusedSubqueryExec => rs.child.id
          }
          assert(subqueriesId.distinct.size == 1)
      }
    )
  }

  test("GLUTEN-3534: Fix incorrect logic of judging whether supports pre-project for the shuffle") {
    withSQLConf(("spark.sql.autoBroadcastJoinThreshold", "-1")) {
      runQueryAndCompare(
        s"""
           |select t1.l_orderkey, t2.o_orderkey, extract(year from t1.l_shipdate), t2.o_year,
           |t1.l_cnt, t2.o_cnt
           |from (
           |  select l_orderkey, l_shipdate, count(1) as l_cnt
           |  from lineitem
           |  group by l_orderkey, l_shipdate) t1
           |join (
           |  select o_orderkey, extract(year from o_orderdate) as o_year, count(1) as o_cnt
           |  from orders
           |  group by o_orderkey, o_orderdate) t2
           |on t1.l_orderkey = t2.o_orderkey
           | and extract(year from t1.l_shipdate) = o_year
           |order by t1.l_orderkey, t2.o_orderkey, t2.o_year, t1.l_cnt, t2.o_cnt
           |limit 100
           |
           |""".stripMargin
      )(df => {})

      runQueryAndCompare(
        s"""
           |select t1.l_orderkey, t2.o_orderkey, extract(year from t1.l_shipdate), t2.o_year
           |from (
           |  select l_orderkey, l_shipdate, count(1) as l_cnt
           |  from lineitem
           |  group by l_orderkey, l_shipdate) t1
           |join (
           |  select o_orderkey, extract(year from o_orderdate) as o_year, count(1) as o_cnt
           |  from orders
           |  group by o_orderkey, o_orderdate) t2
           |on t1.l_orderkey = t2.o_orderkey
           | and extract(year from t1.l_shipdate) = o_year
           |order by t1.l_orderkey, t2.o_orderkey, t2.o_year
           |limit 100
           |
           |""".stripMargin)(df => {})
    }
  }

  test("GLUTEN-7673: fix substrait infinite loop") {
    withSQLConf(("spark.sql.autoBroadcastJoinThreshold", "-1")) {
      val result = sql(
        s"""
           |select l_orderkey
           |from lineitem
           |inner join orders
           |on l_orderkey = o_orderkey
           |  and ((l_shipdate = '2024-01-01' and l_partkey=1
           |  and l_suppkey>2 and o_orderpriority=-987)
           |  or l_shipmode>o_comment)
           |order by l_orderkey limit 1
           |""".stripMargin
      ).collect()
      // check no exception
      assert(result.length == 1)
    }
  }
}
