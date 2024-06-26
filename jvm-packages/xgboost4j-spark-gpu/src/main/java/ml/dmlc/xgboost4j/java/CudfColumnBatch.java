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

import java.util.ArrayList;
import java.util.List;

import ai.rapids.cudf.ColumnVector;
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

  private List<CudfColumn> features;
  private CudfColumn label;
  private CudfColumn weight;
  private CudfColumn baseMargin;

  public CudfColumnBatch(Table featureTable, Table labelTable, Table weightTable,
                         Table baseMarginTable) {
    this.featureTable = featureTable;
    this.labelTable = labelTable;
    this.weightTable = weightTable;
    this.baseMarginTable = baseMarginTable;

    features = new ArrayList<>();
    for (int index = 0; index < featureTable.getNumberOfColumns(); index++) {
      ColumnVector cv = featureTable.getColumn(index);
      features.add(CudfColumn.from(cv));
    }

    if (labelTable != null) {
      assert labelTable.getNumberOfColumns() == 1;
      label = CudfColumn.from(labelTable.getColumn(0));
    }

    if (weightTable != null) {
      assert weightTable.getNumberOfColumns() == 1;
      weight = CudfColumn.from(weightTable.getColumn(0));
    }

    // TODO baseMargin should be an array for multi classification
    if (baseMarginTable != null) {
      assert baseMarginTable.getNumberOfColumns() == 1;
      baseMargin = CudfColumn.from(baseMarginTable.getColumn(0));
    }
  }

  public List<CudfColumn> getFeatures() {
    return features;
  }

  public CudfColumn getLabel() {
    return label;
  }

  public CudfColumn getWeight() {
    return weight;
  }

  public CudfColumn getBaseMargin() {
    return baseMargin;
  }

  public String toJson() {
    ObjectMapper mapper = new ObjectMapper();
    mapper.setSerializationInclusion(JsonInclude.Include.NON_NULL);
    String json = "";
    try {
      json = mapper.writeValueAsString(this);
    } catch (JsonProcessingException e) {
      throw new RuntimeException(e);
    }
    return json;
  }

  @Override
  public void close() {
    if (featureTable != null) featureTable.close();
    if (labelTable != null) labelTable.close();
    if (weightTable != null) weightTable.close();
    if (baseMarginTable != null) baseMarginTable.close();
  }
}
