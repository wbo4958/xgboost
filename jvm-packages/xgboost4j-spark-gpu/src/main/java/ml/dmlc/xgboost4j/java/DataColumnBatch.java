package ml.dmlc.xgboost4j.java;

import com.fasterxml.jackson.annotation.JsonInclude;

import java.util.ArrayList;
import java.util.List;

@JsonInclude(JsonInclude.Include.NON_NULL)
class DataColumn {
  public List<Long> shape = new ArrayList<>();   // row count
  public List<String> data = new ArrayList<>(); //  gpu data buffer address
  public String typestr;
  public int version = 1;
  public DataColumn mask = null;

  public DataColumn(long shape, long data, String typestr, int version, DataColumn mask) {
    this.shape.add(shape);
    this.data.add(data + ",false");
    this.typestr = typestr;
    this.version = version;
    this.mask = mask;
  }
}

public class DataColumnBatch implements ColumnBatch {

  private final List<DataColumn> features;
  private final DataColumn labels;
  private final DataColumn weights;
  private final DataColumn baseMargins;

  public DataColumnBatch(List<DataColumn> features, DataColumn labels, DataColumn weights, DataColumn baseMargins) {
    this.features = features;
    this.labels = labels;
    this.weights = weights;
    this.baseMargins = baseMargins;
  }

  @Override
  public String getFeatureArrayInterface() {
    return "";
  }

  @Override
  public String getLabelsArrayInterface() {
    return "";
  }

  @Override
  public String getWeightsArrayInterface() {
    return "";
  }

  @Override
  public String getBaseMarginsArrayInterface() {
    return "";
  }
}
