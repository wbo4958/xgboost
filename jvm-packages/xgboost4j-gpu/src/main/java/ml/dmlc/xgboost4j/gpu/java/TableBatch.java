/*
 Copyright (c) 2021 by Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */

package ml.dmlc.xgboost4j.gpu.java;

import java.util.Iterator;

/**
 * A mini-batch of Table that can be converted to GpuDMatrix.
 *
 * This class is used to support advanced creation of DMatrix from Iterator of TableBatch,
 */
public class TableBatch implements AutoCloseable {

  private String arrayInterfaceJson;
  private GpuTable table;

  public TableBatch(GpuTable table, String arrayInterfaceJson) {
    this.table = table;
    this.arrayInterfaceJson = arrayInterfaceJson;
  }

  // Called from native
  public String getArrayInterfaceJson() {
    return arrayInterfaceJson;
  }

  // Called from native
  @Override
  public void close() {
    table.close();
  }

  public static class ColumnBatchIterator implements Iterator<TableBatch> {
    private Iterator<GpuTable> base;

    public ColumnBatchIterator(Iterator<GpuTable> base) {
      this.base = base;
    }

    @Override
    public boolean hasNext() {
      return base.hasNext();
    }

    @Override
    public TableBatch next() {
      GpuTable table = base.next();
      return new TableBatch(table, table.getJson());
    }
  }
}
