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
package org.apache.gluten.vectorized;

import io.github.zhztheplayer.velox4j.data.RowVector;

import java.util.ArrayDeque;
import java.util.Iterator;

/** Iterator for velox RowVector. */
public class VLVectorIterator implements Iterator<RowVector> {

  private final ArrayDeque<RowVector> rows;

  public VLVectorIterator() {
    this.rows = new ArrayDeque<>();
  }

  public boolean hasNext() {
    return !rows.isEmpty();
  }

  public RowVector next() {
    if (!hasNext()) {
      return null;
    }
    return rows.removeFirst();
  }

  public void addRow(RowVector row) {
    rows.add(row);
  }
}
