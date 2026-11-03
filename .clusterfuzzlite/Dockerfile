FROM gcr.io/oss-fuzz-base/base-builder-rust

COPY . $SRC/repose
# The ClusterFuzzLite builder invokes $SRC/build.sh inside the image.
COPY .clusterfuzzlite/build.sh $SRC/build.sh
WORKDIR $SRC/repose
