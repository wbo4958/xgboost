/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class ml_dmlc_xgboost4j_gpu_java_GpuXGBoostJNI */

#ifndef _Included_ml_dmlc_xgboost4j_gpu_java_GpuXGBoostJNI
#define _Included_ml_dmlc_xgboost4j_gpu_java_GpuXGBoostJNI
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     ml_dmlc_xgboost4j_gpu_java_GpuXGBoostJNI
 * Method:    XGBGetLastError
 * Signature: ()Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_ml_dmlc_xgboost4j_gpu_java_XGBoostJNI_XGBGpuGetLastError
  (JNIEnv *, jclass);

/*
 * Class:     ml_dmlc_xgboost4j_gpu_java_GpuXGBoostJNI
 * Method:    XGDMatrixSetInfoFromInterface
 * Signature: (JLjava/lang/String;Ljava/lang/String;)I
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_gpu_java_XGBoostJNI_XGDMatrixSetInfoFromInterface
  (JNIEnv *, jclass, jlong, jstring, jstring);

/*
 * Class:     ml_dmlc_xgboost4j_gpu_java_GpuXGBoostJNI
 * Method:    XGDeviceQuantileDMatrixCreateFromCallback
 * Signature: (Ljava/util/Iterator;FII[J)I
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_gpu_java_XGBoostJNI_XGDeviceQuantileDMatrixCreateFromCallback
  (JNIEnv *, jclass, jobject, jfloat, jint, jint, jlongArray);

/*
 * Class:     ml_dmlc_xgboost4j_gpu_java_GpuXGBoostJNI
 * Method:    XGDMatrixCreateFromArrayInterfaceColumns
 * Signature: (Ljava/lang/String;FI[J)I
 */
JNIEXPORT jint JNICALL Java_ml_dmlc_xgboost4j_gpu_java_XGBoostJNI_XGDMatrixCreateFromArrayInterfaceColumns
(JNIEnv *, jclass, jstring, jfloat, jint, jlongArray);

#ifdef __cplusplus
}
#endif
#endif
