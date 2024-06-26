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

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import ai.rapids.cudf.Table;
import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;

/**
 * CudfColumnBatch wraps multiple CudfColumns to provide the cuda
 * array interface json string for all columns.
 */
public class CudfColumnBatch extends ColumnBatch {
  @JsonIgnore
  private final Table featureTable;
  @JsonIgnore
  private final Table labelTable;
  @JsonIgnore
  private final Table weightTable;
  @JsonIgnore
  private final Table baseMarginTable;

  private List<CudfColumn> features_str;
  private List<CudfColumn> label_str;
  private List<CudfColumn> weight_str;
  private List<CudfColumn> basemargin_str;

  public CudfColumnBatch(Table featureTable, Table labelTable, Table weightTable,
                         Table baseMarginTable) {
    this.featureTable = featureTable;
    this.labelTable = labelTable;
    this.weightTable = weightTable;
    this.baseMarginTable = baseMarginTable;

    features_str = initializeCudfColumns(featureTable);
    if (labelTable != null) {
      assert labelTable.getNumberOfColumns() == 1;
      label_str = initializeCudfColumns(labelTable);
    }

    if (weightTable != null) {
      assert weightTable.getNumberOfColumns() == 1;
      weight_str = initializeCudfColumns(weightTable);
    }

    if (baseMarginTable != null) {
      basemargin_str = initializeCudfColumns(baseMarginTable);
    }
  }

  private List<CudfColumn> initializeCudfColumns(Table table) {
    assert table != null && table.getNumberOfColumns() > 0;

    return IntStream.range(0, table.getNumberOfColumns())
      .mapToObj(table::getColumn)
      .map(CudfColumn::from)
      .collect(Collectors.toList());
  }

  public List<CudfColumn> getFeatures_str() {
    return features_str;
  }

  public List<CudfColumn> getLabel_str() {
    return label_str;
  }

  public List<CudfColumn> getWeight_str() {
    return weight_str;
  }

  public List<CudfColumn> getBasemargin_str() {
    return basemargin_str;
  }

  public String toJson() {
    ObjectMapper mapper = new ObjectMapper();
    mapper.setSerializationInclusion(JsonInclude.Include.NON_NULL);
    try {
      return mapper.writeValueAsString(this);
    } catch (JsonProcessingException e) {
      throw new RuntimeException(e);
    }
  }

  @Override
  public String toFeaturesJson() {
    ObjectMapper mapper = new ObjectMapper();
    try {
      return mapper.writeValueAsString(features_str);
    } catch (JsonProcessingException e) {
      throw new RuntimeException(e);
    }
  }

  @Override
  public void close() {
    if (featureTable != null) featureTable.close();
    if (labelTable != null) labelTable.close();
    if (weightTable != null) weightTable.close();
    if (baseMarginTable != null) baseMarginTable.close();
  }
}
