#! /bin/bash


files=`find include -type f`
files=`find R-package -type f`
files=`find python-package -type f`

for x in include R-package python-package rabit; do
  echo $x
  files=`find $x -type f`
  for file in $files
  do
    diff /home/bobwang/work.d/nvspark/xgboost/dmlc.xgboost/xgboost/$file /home/bobwang/work.d/nvspark/xgboost/nvmain.xgboost/xgboost/$file
    if [ $? != 0 ]; then
      echo "$file"
    fi
  done
done

