FROM gcr.io/oss-fuzz-base/base-builder-rust

# The pinned builder image ships a nightly older than the rust-version
# required by cu29 (1.95); refresh it before building the fuzz targets.
# RUSTUP_TOOLCHAIN is pinned in the base image, so repoint it explicitly.
RUN rustup toolchain install nightly
ENV RUSTUP_TOOLCHAIN=nightly

COPY . $SRC/repose
# The ClusterFuzzLite builder invokes $SRC/build.sh inside the image.
COPY .clusterfuzzlite/build.sh $SRC/build.sh
WORKDIR $SRC/repose
