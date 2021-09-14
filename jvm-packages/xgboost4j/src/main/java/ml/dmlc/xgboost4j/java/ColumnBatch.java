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

package ml.dmlc.xgboost4j.java;

import java.util.Iterator;

/**
 * The abstracted XGBoost ColumnBatch to get cuda array interface which is used to build
 * the DMatrix on device.
 *
 */
public abstract class ColumnBatch implements AutoCloseable {

  /**
   * Get the cuda array interface json string for the whole ColumnBatch including
   * the must-have feature, label columns and the optional weight, base margin columns.
   *
   * This API will be called by {@link DMatrix#DMatrix(Iterator, float, int, int)}
   *
   */
  public abstract String getArrayInterfaceJson();

  /**
   * Get the cuda array interface of the feature columns.
   *
   * This API will be called by {@link DMatrix#DMatrix(ColumnBatch, float, int)}
   */
  public abstract String getFeatureArrayInterface();

  @Override
  public void close() throws Exception {}
}
