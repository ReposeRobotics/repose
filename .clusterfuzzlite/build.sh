#!/bin/bash -eu
# Build the repose-kinematics fuzz targets and copy them to $OUT.
# The fuzz crate lives inside the library crate; cargo-fuzz finds it there.

cd $SRC/repose/components/libs/repose_kinematics
cargo fuzz build -O
cp fuzz/target/x86_64-unknown-linux-gnu/release/ik $OUT/
