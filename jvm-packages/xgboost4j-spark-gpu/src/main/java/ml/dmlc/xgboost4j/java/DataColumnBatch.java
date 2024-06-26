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

import ai.rapids.cudf.BaseDeviceMemoryBuffer;
import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.DType;
import ai.rapids.cudf.Table;
import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.util.ArrayList;
import java.util.List;

@JsonInclude(JsonInclude.Include.NON_NULL)
class DataColumn extends Column {
  private List<Long> shape = new ArrayList<>();   // row count
  private List<String> data = new ArrayList<>(); //  gpu data buffer address
  private String typestr;
  private int version = 1;
  private DataColumn mask = null;

  public DataColumn(long shape, long data, String typestr, int version) {
    this.shape.add(shape);
    this.data.add(data + ",false");
    this.typestr = typestr;
    this.version = version;
  }

  public List<Long> getShape() {
    return shape;
  }

  public List<String> getData() {
    return data;
  }

  public String getTypestr() {
    return typestr;
  }

  public int getVersion() {
    return version;
  }

  public DataColumn getMask() {
    return mask;
  }

  public void setMask(DataColumn mask) {
    this.mask = mask;
  }

  @Override
  public String toJson() {
    ObjectMapper mapper = new ObjectMapper();
    mapper.setSerializationInclusion(JsonInclude.Include.NON_NULL);
    try {
      return mapper.writeValueAsString(this);
    } catch (JsonProcessingException e) {
      throw new RuntimeException(e);
    }
  }

    /**
   * Create DataColumn according to ColumnVector
   */
  public static DataColumn from(ColumnVector cv) {
    BaseDeviceMemoryBuffer dataBuffer = cv.getData();
    assert dataBuffer != null;

    DType dType = cv.getType();
    String typeStr = "";
    if (dType == DType.FLOAT32 || dType == DType.FLOAT64 ||
      dType == DType.TIMESTAMP_DAYS || dType == DType.TIMESTAMP_MICROSECONDS ||
      dType == DType.TIMESTAMP_MILLISECONDS || dType == DType.TIMESTAMP_NANOSECONDS ||
      dType == DType.TIMESTAMP_SECONDS) {
      typeStr = "<f" + dType.getSizeInBytes();
    } else if (dType == DType.BOOL8 || dType == DType.INT8 || dType == DType.INT16 ||
      dType == DType.INT32 || dType == DType.INT64) {
      typeStr = "<i" + dType.getSizeInBytes();
    } else {
      // Unsupported type.
      throw new IllegalArgumentException("Unsupported data type: " + dType);
    }

    DataColumn data = new DataColumn(cv.getRowCount(), dataBuffer.getAddress(), typeStr, 1);

    BaseDeviceMemoryBuffer validBuffer = cv.getValid();
    if (validBuffer != null && cv.getNullCount() != 0) {
      DataColumn mask = new DataColumn(cv.getRowCount(), validBuffer.getAddress(), "<t1", 1);
      data.setMask(mask);
    }
    return data;
  }

}

public class DataColumnBatch implements ColumnBatch {
  @JsonIgnore
  private final Table featureTable;
  @JsonIgnore
  private final Table labelTable;
  @JsonIgnore
  private final Table weightTable;
  @JsonIgnore
  private final Table baseMarginTable;

  private List<DataColumn> features;
  private DataColumn label;
  private DataColumn weight;
  private DataColumn baseMargin;

  public DataColumnBatch(Table featureTable, Table labelTable, Table weightTable, Table baseMarginTable) {
    this.featureTable = featureTable;
    this.labelTable = labelTable;
    this.weightTable = weightTable;
    this.baseMarginTable = baseMarginTable;

    features = new ArrayList<>();
    for (int index = 0; index < featureTable.getNumberOfColumns(); index++) {
      ColumnVector cv = featureTable.getColumn(index);
      features.add(DataColumn.from(cv));
    }

    if (labelTable != null) {
      assert labelTable.getNumberOfColumns() == 1;
      label = DataColumn.from(labelTable.getColumn(0));
    }

    if (weightTable != null) {
      assert weightTable.getNumberOfColumns() == 1;
      weight = DataColumn.from(weightTable.getColumn(0));
    }

    // TODO baseMargin should be an array for multi classification
    if (baseMarginTable != null) {
      assert baseMarginTable.getNumberOfColumns() == 1;
      baseMargin = DataColumn.from(baseMarginTable.getColumn(0));
    }
  }

  public List<DataColumn> getFeatures() {
    return features;
  }

  public DataColumn getLabel() {
    return label;
  }

  public DataColumn getWeight() {
    return weight;
  }

  public DataColumn getBaseMargin() {
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
