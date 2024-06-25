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

import ai.rapids.cudf.*;

/**
 * This class is composing of base data with Apache Arrow format from Cudf ColumnVector.
 * It will be used to generate the cuda array interface.
 */
public class CudfColumn extends Column {

  private final long dataPtr; //  gpu data buffer address
  private final long shape;   // row count
  private final long validPtr; // gpu valid buffer address
  private final long offsetPtr; // gpu offset address for list
  private final int typeSize; // type size in bytes
  private final String typeStr; // follow array interface spec
  private final long nullCount; // null count

  private String arrayInterface = null; // the cuda array interface

  public static CudfColumn from(ColumnVector column) {
    ColumnView cv = column;
    DType dType = cv.getType();
    long nullCount = cv.getNullCount();
    String floatType = "<f";
    String integerType = "<i";
    long offsetPtr = 0;
    if (dType == DType.LIST) {
      floatType = "<lf";
      integerType = "<li";
      if (cv.getOffsets() != null) {
        offsetPtr = cv.getOffsets().getAddress();
      }
      cv = cv.getChildColumnView(0);
      dType = cv.getType();
    }

    BaseDeviceMemoryBuffer dataBuffer = cv.getData();
    BaseDeviceMemoryBuffer validBuffer = cv.getValid();
    long validPtr = 0;
    if (validBuffer != null) {
      validPtr = validBuffer.getAddress();
    }
    String typeStr = "";
    if (dType == DType.FLOAT32 || dType == DType.FLOAT64 ||
      dType == DType.TIMESTAMP_DAYS || dType == DType.TIMESTAMP_MICROSECONDS ||
      dType == DType.TIMESTAMP_MILLISECONDS || dType == DType.TIMESTAMP_NANOSECONDS ||
      dType == DType.TIMESTAMP_SECONDS) {
      typeStr = floatType + dType.getSizeInBytes();
    } else if (dType == DType.BOOL8 || dType == DType.INT8 || dType == DType.INT16 ||
      dType == DType.INT32 || dType == DType.INT64) {
      typeStr = integerType + dType.getSizeInBytes();
    } else {
      // Unsupported type.
      throw new IllegalArgumentException("Unsupported data type: " + dType);
    }

    return new CudfColumn(dataBuffer.getAddress(), cv.getRowCount(), validPtr,
                          dType.getSizeInBytes(), typeStr, nullCount, offsetPtr);
  }

  private CudfColumn(long dataPtr, long shape, long validPtr, int typeSize, String typeStr,
                     long nullCount, long offsetPtr) {
    this.dataPtr = dataPtr;
    this.shape = shape;
    this.validPtr = validPtr;
    this.typeSize = typeSize;
    this.typeStr = typeStr;
    this.nullCount = nullCount;
    this.offsetPtr = offsetPtr;
  }

  @Override
  public String getArrayInterfaceJson() {
    // There is no race-condition
    if (arrayInterface == null) {
      arrayInterface = CudfUtils.buildArrayInterface(this);
    }
    return arrayInterface;
  }

  public long getDataPtr() {
    return dataPtr;
  }

  public long getShape() {
    return shape;
  }

  public long getValidPtr() {
    return validPtr;
  }

  public int getTypeSize() {
    return typeSize;
  }

  public String getTypeStr() {
    return typeStr;
  }

  public long getNullCount() {
    return nullCount;
  }

}
