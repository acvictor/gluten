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
package org.apache.spark.sql.sources

import org.apache.gluten.execution.{FileSourceScanExecTransformer, ShuffledHashJoinExecTransformer}

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.GlutenTestsCommonTrait
import org.apache.spark.sql.execution.{ColumnarShuffleExchangeExec, FileSourceScanExec, FileSourceScanLike, SparkPlan}
import org.apache.spark.sql.execution.adaptive.{AdaptiveSparkPlanExec, ShuffleQueryStageExec}
import org.apache.spark.sql.execution.exchange.ShuffleExchangeExec
import org.apache.spark.sql.internal.SQLConf

class GlutenBucketedWriteWithHiveSupportSuite
  extends BucketedWriteWithHiveSupportSuite
  with GlutenTestsCommonTrait {
  override def testBucketingCondition(
      shuffleLeft: Boolean,
      sortLeft: Boolean,
      numOutputPartitionsLeft: Option[Int],
      shuffleRight: Boolean,
      sortRight: Boolean,
      numOutputPartitionsRight: Option[Int],
      joined: DataFrame): Unit = {
    joined.collect()
    val executedPlan = if (spark.conf.get(SQLConf.ADAPTIVE_EXECUTION_ENABLED)) {
      joined.queryExecution.executedPlan.asInstanceOf[AdaptiveSparkPlanExec].executedPlan
    } else {
      joined.queryExecution.executedPlan
    }
    val joinOperator = {
      val shuffleExec = executedPlan.collect { case s: ShuffledHashJoinExecTransformer => s }
      assert(shuffleExec.size == 1)
      shuffleExec.head
    }

    // check existence of shuffle
    assert(
      containsShuffleExchangeExec(joinOperator.left) == shuffleLeft,
      s"expected shuffle in plan to be $shuffleLeft but found\n${joinOperator.left}"
    )
    assert(
      containsShuffleExchangeExec(joinOperator.right) == shuffleRight,
      s"expected shuffle in plan to be $shuffleRight but found\n${joinOperator.right}"
    )

    // check the output partitioning
    if (numOutputPartitionsLeft.isDefined) {
      assert(
        joinOperator.left.outputPartitioning.numPartitions ===
          numOutputPartitionsLeft.get)
    }
    if (numOutputPartitionsRight.isDefined) {
      assert(
        joinOperator.right.outputPartitioning.numPartitions ===
          numOutputPartitionsRight.get)
    }
  }

  def containsShuffleExchangeExec(plan: SparkPlan): Boolean = {
    plan match {
      case _: ColumnarShuffleExchangeExec => true
      case shuffleQ: ShuffleQueryStageExec => containsShuffleExchangeExec(shuffleQ.plan)
      case _ => plan.children.exists(containsShuffleExchangeExec)
    }
  }

  override def getFileScan(plan: SparkPlan): FileSourceScanLike = {
    val fileScan = plan.collect {
      case f: FileSourceScanExec => f
      case nf: FileSourceScanExecTransformer => nf
    }
    assert(fileScan.nonEmpty, plan)
    fileScan.head
  }

  override def verify(
      query: String,
      expectedNumShuffles: Int,
      expectedCoalescedNumBuckets: Option[Int]): Unit = {
    Seq(true, false).foreach {
      aqeEnabled =>
        withSQLConf(SQLConf.ADAPTIVE_EXECUTION_ENABLED.key -> aqeEnabled.toString) {
          val plan = sql(query).queryExecution.executedPlan
          val shuffles = plan.collect {
            case s: ShuffleExchangeExec => s
            case ns: ColumnarShuffleExchangeExec => ns
          }
          assert(shuffles.length == expectedNumShuffles)

          val scans = plan.collect {
            case f: FileSourceScanExec if f.optionalNumCoalescedBuckets.isDefined => f
            case nf: FileSourceScanExecTransformer if nf.optionalNumCoalescedBuckets.isDefined => nf
          }
          if (expectedCoalescedNumBuckets.isDefined) {
            assert(scans.length == 1)
            assert(scans.head.optionalNumCoalescedBuckets == expectedCoalescedNumBuckets)
          } else {
            assert(scans.isEmpty)
          }
        }
    }
  }
}
