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

import java.util.ArrayList;
import java.util.List;

/**
 * A Tensor to hold the prediction result.
 */
public class Tensor {
  // Dimension of Tensor
  private final long dim;
  // Shape of Tensor
  private final long[] shape;
  // The raw result predicted by XGBoost
  private final float[] result;
  // The List type result for Java
  private List resultList;

  public Tensor(long dim, long[] shape, float[] result) {
    this.dim = dim;
    this.shape = shape;
    this.result = result;
  }

  /**
   * Get the dimension of Tensor
   */
  public long getDim() {
    return dim;
  }

  /**
   * Get the shape of Tensor
   */
  public long[] getShape() {
    return shape;
  }

  /**
   * Get the raw prediction result
   */
  public float[] getRawResult() {
    return result;
  }

  /**
   * Convert the raw result to the List type
   */
  public synchronized List getResultList() {
    if (resultList == null) {
      // Is there a better way to construct the multi-dimension result ?
      resultList = new ArrayList();
      for (int i = 0; i < shape[0]; i++) {
        if (dim == 1) {
          resultList.add(result[i]);
          continue;
        }

        ArrayList list1 = new ArrayList();
        for (int j = 0; j < shape[1]; j++) {
          if (dim == 2) {
            int index = (int) (i * shape[1] + j);
            list1.add(result[index]);
            continue;
          }

          ArrayList list2 = new ArrayList();
          for (int k = 0; k < shape[2]; k++) {
            if (dim == 3) {
              int index = (int) (i * shape[1] + j * shape[2] + k);
              list2.add(result[index]);
              continue;
            }

            ArrayList list3 = new ArrayList();
            for (int m = 0; m < shape[3]; m++) {
              int index = (int) (i * shape[1] + j * shape[2] + k * shape[3] + m);
              list3.add(result[index]);
            }
            list2.add(list3);
          }
          list1.add(list2);
        }
        resultList.add(list1);
      }
    }
    return resultList;
  }
}
