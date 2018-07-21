#!/bin/bash

for file in *.png; do
  path=$PWD/out/$(echo $file | sed s#__#/#g | sed s#⊛#x#g )
  dir=$(dirname $path)
  [ ! -d $dir ] && mkdir -p $dir
  mv $file $path
  echo $file "-->" $path
done

