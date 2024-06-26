/*
 Copyright (c) 2021-2024 by Contributors

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

/**
 * The abstracted XGBoost ColumnBatch to get array interface from columnar data format.
 * For example, the cuDF dataframe which employs apache arrow specification.
 */
public abstract class ColumnBatch extends Column {
  /**
   * Get the cuda array interface json string for the whole ColumnBatch including
   * the must-have feature, label columns and the optional weight, base margin columns.
   * <p>
   * This function is be called by native code during iteration and can be made as private
   * method.  We keep it as public simply to silent the linter.
   */
}
