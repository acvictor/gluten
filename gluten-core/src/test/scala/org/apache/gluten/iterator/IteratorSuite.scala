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
package org.apache.gluten.iterator

import org.apache.gluten.exception.GlutenException
import org.apache.gluten.iterator.Iterators.{V1, WrapperBuilder}

import org.apache.spark.task.TaskResources

import org.scalatest.funsuite.AnyFunSuite

class IteratorV1Suite extends IteratorSuite {
  override protected def wrap[A](in: Iterator[A]): WrapperBuilder[A] = Iterators.wrap(V1, in)
}

abstract class IteratorSuite extends AnyFunSuite {
  protected def wrap[A](in: Iterator[A]): WrapperBuilder[A]

  test("Trivial wrapping") {
    val strings = Array[String]("one", "two", "three")
    val itr = strings.toIterator
    val wrapped = wrap(itr)
      .create()
    assertResult(strings) {
      wrapped.toArray
    }
  }

  test("Complete iterator") {
    var completeCount = 0
    TaskResources.runUnsafe {
      val strings = Array[String]("one", "two", "three")
      val itr = strings.toIterator
      val wrapped = wrap(itr)
        .recycleIterator {
          completeCount += 1
        }
        .create()
      assertResult(strings) {
        wrapped.toArray
      }
      assert(completeCount == 1)
    }
    assert(completeCount == 1)
  }

  test("Complete intermediate iterator") {
    var completeCount = 0
    TaskResources.runUnsafe {
      val strings = Array[String]("one", "two", "three")
      val itr = strings.toIterator
      val _ = wrap(itr)
        .recycleIterator {
          completeCount += 1
        }
        .create()
      assert(completeCount == 0)
    }
    assert(completeCount == 1)
  }

  test("Close payload") {
    var closeCount = 0
    TaskResources.runUnsafe {
      val strings = Array[String]("one", "two", "three")
      val itr = strings.toIterator
      val wrapped = wrap(itr)
        .recyclePayload { _: String => closeCount += 1 }
        .create()
      assertResult(strings) {
        wrapped.toArray
      }
      assert(closeCount == 3)
    }
    assert(closeCount == 3)
  }

  test("Close intermediate payload") {
    var closeCount = 0
    TaskResources.runUnsafe {
      val strings = Array[String]("one", "two", "three")
      val itr = strings.toIterator
      val wrapped = wrap(itr)
        .recyclePayload { _: String => closeCount += 1 }
        .create()
      assertResult(strings.take(2)) {
        wrapped.take(2).toArray
      }
      assert(closeCount == 1) // the first one is closed after consumed
    }
    assert(closeCount == 2) // the second one is closed on task exit
  }

  test("Protect invocation flow") {
    var hasNextCallCount = 0
    var nextCallCount = 0
    val itr = new Iterator[Any] {
      override def hasNext: Boolean = {
        hasNextCallCount += 1
        true
      }

      override def next(): Any = {
        nextCallCount += 1
        new Object
      }
    }
    val wrapped = wrap(itr)
      .protectInvocationFlow()
      .create()
    wrapped.hasNext
    assert(hasNextCallCount == 1)
    assert(nextCallCount == 0)
    wrapped.hasNext
    assert(hasNextCallCount == 1)
    assert(nextCallCount == 0)
    wrapped.next
    assert(hasNextCallCount == 1)
    assert(nextCallCount == 1)
    wrapped.next
    assert(hasNextCallCount == 2)
    assert(nextCallCount == 2)
  }
}

/**
 * Tests that ClosableIterator preserves RuntimeException subclasses as is and wraps checked
 * exceptions in GlutenException.
 */
class ClosableIteratorSuite extends AnyFunSuite {

  test("RuntimeException from hasNext0 propagates as-is") {
    val ex = new ArithmeticException("overflow")
    val itr = new ClosableIterator[Int] {
      override protected def hasNext0(): Boolean = throw ex
      override protected def next0(): Int = 0
      override protected def close0(): Unit = {}
    }
    val caught = intercept[ArithmeticException](itr.hasNext)
    assert(caught eq ex)
  }

  test("RuntimeException from next0 propagates as-is") {
    val ex = new IllegalStateException("bad state")
    val itr = new ClosableIterator[Int] {
      override protected def hasNext0(): Boolean = true
      override protected def next0(): Int = throw ex
      override protected def close0(): Unit = {}
    }
    val caught = intercept[IllegalStateException](itr.next())
    assert(caught eq ex)
  }

  test("GlutenException from hasNext0 propagates as-is") {
    val ex = new GlutenException("gluten error")
    val itr = new ClosableIterator[Int] {
      override protected def hasNext0(): Boolean = throw ex
      override protected def next0(): Int = 0
      override protected def close0(): Unit = {}
    }
    val caught = intercept[GlutenException](itr.hasNext)
    assert(caught eq ex)
  }

  test("Checked exception from hasNext0 is wrapped in GlutenException") {
    val ioEx = new java.io.IOException("io error")
    val itr = new ClosableIterator[Int] {
      override protected def hasNext0(): Boolean = throw ioEx
      override protected def next0(): Int = 0
      override protected def close0(): Unit = {}
    }
    val caught = intercept[GlutenException](itr.hasNext)
    assert(caught.getCause eq ioEx)
  }

  test("Checked exception from next0 is wrapped in GlutenException") {
    val ioEx = new java.io.IOException("io error")
    val itr = new ClosableIterator[Int] {
      override protected def hasNext0(): Boolean = true
      override protected def next0(): Int = throw ioEx
      override protected def close0(): Unit = {}
    }
    val caught = intercept[GlutenException](itr.next())
    assert(caught.getCause eq ioEx)
  }
}
