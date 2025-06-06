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
package org.apache.gluten.test

import org.apache.gluten.execution.GlutenPlan

import org.apache.spark.internal.Logging
import org.apache.spark.sql.execution._
import org.apache.spark.sql.execution.adaptive.{AdaptiveSparkPlanExec, AdaptiveSparkPlanHelper, AQEShuffleReadExec, QueryStageExec}
import org.apache.spark.sql.execution.exchange.ReusedExchangeExec

/**
 * attention: if AQE is enabled,This method will only be executed correctly after the execution plan
 * is fully determined
 */

object FallbackUtil extends Logging with AdaptiveSparkPlanHelper {

  def tolerate(plan: SparkPlan): Boolean = {
    plan match {
      case _: ColumnarToRowTransition =>
        true
      case _: RowToColumnarTransition =>
        true
      case _: BaseSubqueryExec =>
        true
      case _: QueryStageExec =>
        true
      case WholeStageCodegenExec(_) =>
        true
      case ColumnarInputAdapter(_) =>
        true
      case InputAdapter(_) =>
        true
      case AdaptiveSparkPlanExec(_, _, _, _, _) =>
        true
      case AQEShuffleReadExec(_, _) =>
        true
      case _: LimitExec =>
        true
      // for ut
      case _: RangeExec =>
        true
      case _: ObjectConsumerExec =>
        true
      case _: LocalTableScanExec =>
        true
      case _: ReusedExchangeExec =>
        true
      case _ =>
        false
    }
  }

  def hasFallback(plan: SparkPlan): Boolean = {
    val fallbackOperator =
      collectWithSubqueries(plan) { case plan => plan }.filter(plan => nodeHasFallback(plan))
    fallbackOperator.foreach(operator => log.info(s"gluten fallback operator:{$operator}"))
    fallbackOperator.nonEmpty
  }

  def nodeHasFallback(plan: SparkPlan): Boolean = {
    !plan.isInstanceOf[GlutenPlan] && !tolerate(plan)
  }
}
