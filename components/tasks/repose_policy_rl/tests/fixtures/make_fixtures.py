#!/usr/bin/env python3
"""Build the tiny ONNX fixtures committed next to this script.

The generated `.onnx` files are committed, so the Rust tests need no Python.
Regenerate after editing this file with:

    uv run --with onnx --with numpy python3 make_fixtures.py

Fixtures:
  identity_4.onnx   output = input, [1, 4] -> [1, 4]
  linear_24_6.onnx  one Gemm: output = GAIN * goal_error slice of the
                    24-wide observation (joint_positions 6, joint_velocities
                    6, goal_error 6, previous_action 6), [1, 24] -> [1, 6].
                    A proportional controller written as a network.
  bad_shape.onnx    Gemm whose weight needs a 5-wide input, [1, 5] -> [1, 4];
                    loading it for OBS = 4 must fail at startup.
  nan.onnx          output = input + NaN, [1, 4] -> [1, 4]; drives the Fault
                    path (non-finite model output).
  double_4.onnx     identity_4 with float64 tensors; loading it must fail at
                    startup because the model input is not f32.
"""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

GAIN = 0.5
OPSET = 13
IR_VERSION = 8


def save(model, name):
    model.ir_version = IR_VERSION
    onnx.checker.check_model(model)
    onnx.save(model, name)
    print(f"wrote {name}")


def value_info(name, shape):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def identity_4():
    node = helper.make_node("Identity", ["observation"], ["action"])
    graph = helper.make_graph(
        [node],
        "identity_4",
        [value_info("observation", [1, 4])],
        [value_info("action", [1, 4])],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])


def linear_24_6():
    # Y = X @ W + B with W selecting the goal_error slice (indices 12..18).
    weights = np.zeros((24, 6), dtype=np.float32)
    for i in range(6):
        weights[12 + i, i] = GAIN
    bias = np.zeros(6, dtype=np.float32)
    node = helper.make_node("Gemm", ["observation", "weights", "bias"], ["action"])
    graph = helper.make_graph(
        [node],
        "linear_24_6",
        [value_info("observation", [1, 24])],
        [value_info("action", [1, 6])],
        [
            numpy_helper.from_array(weights, "weights"),
            numpy_helper.from_array(bias, "bias"),
        ],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])


def bad_shape():
    weights = np.zeros((5, 4), dtype=np.float32)
    bias = np.zeros(4, dtype=np.float32)
    node = helper.make_node("Gemm", ["observation", "weights", "bias"], ["action"])
    graph = helper.make_graph(
        [node],
        "bad_shape",
        [value_info("observation", [1, 5])],
        [value_info("action", [1, 4])],
        [
            numpy_helper.from_array(weights, "weights"),
            numpy_helper.from_array(bias, "bias"),
        ],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])


def nan_model():
    offset = np.full((1, 4), np.nan, dtype=np.float32)
    node = helper.make_node("Add", ["observation", "offset"], ["action"])
    graph = helper.make_graph(
        [node],
        "nan",
        [value_info("observation", [1, 4])],
        [value_info("action", [1, 4])],
        [numpy_helper.from_array(offset, "offset")],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])


def double_4():
    node = helper.make_node("Identity", ["observation"], ["action"])
    graph = helper.make_graph(
        [node],
        "double_4",
        [helper.make_tensor_value_info("observation", TensorProto.DOUBLE, [1, 4])],
        [helper.make_tensor_value_info("action", TensorProto.DOUBLE, [1, 4])],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])


if __name__ == "__main__":
    save(identity_4(), "identity_4.onnx")
    save(linear_24_6(), "linear_24_6.onnx")
    save(bad_shape(), "bad_shape.onnx")
    save(nan_model(), "nan.onnx")
    save(double_4(), "double_4.onnx")
