#!/bin/bash

mkdir -p rpm
yumdownloader --assumeyes --destdir rpm --resolve \
  libglvnd-glx.x86_64 \
  mesa-libGL.x86_64\
  mesa-libGL-devel.x86_64\
  mesa-libOSMesa.x86_64\
  mesa-libOSMesa-devel.x86_64 \
  patchelf.x86_64
