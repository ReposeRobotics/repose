#!/usr/bin/env python3
"""Headless MuJoCo transport for the Copper ReBeL example.

Serves N joints (every joint in the URDF, named joint1..jointN). The Copper
bridge must be `CuMujocoSimBridge<N>` with the same N.
"""
import argparse
import json
import socket
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


def load_model(path):
    tree = ET.parse(path)
    # This example uses Copper's collision model and needs no visual meshes.
    for link in tree.getroot().iter('link'):
        for child in list(link):
            if child.tag in ('visual', 'collision'):
                link.remove(child)
    with tempfile.NamedTemporaryFile(suffix='.urdf') as source:
        tree.write(source.name)
        spec = mujoco.MjSpec.from_file(source.name)
    spec.option.timestep = 0.002
    joints = list(spec.joints)
    expected = [f'joint{i}' for i in range(1, len(joints) + 1)]
    if not joints or [joint.name for joint in joints] != expected:
        raise ValueError('URDF must contain joint1 through jointN in order')
    for joint in joints:
        spec.add_actuator(name='motor_' + joint.name,
                          type=mujoco.mjtActuator.mjACTUATOR_MOTOR,
                          trntype=mujoco.mjtTrn.mjTRN_JOINT, target=joint.name,
                          gear=[1, 0, 0, 0, 0, 0], forcelimited=True,
                          forcerange=[-60, 60])
    return spec.compile()


def send_state(conn, model, data):
    n = model.nu
    message = {'type': 'state', 'n': n, 'q': data.qpos[:n].tolist(),
               'qd': data.qvel[:n].tolist(), 'time': float(data.time)}
    conn.sendall((json.dumps(message, allow_nan=False) + '\n').encode())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--urdf', required=True, help='igus ReBeL URDF')
    parser.add_argument('--port', type=int, default=55091)
    args = parser.parse_args()
    model = load_model(args.urdf)
    data = mujoco.MjData(model)
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(('127.0.0.1', args.port))
        server.listen(1)
        print(f'MuJoCo ready on 127.0.0.1:{args.port}', flush=True)
        while True:
            conn, _ = server.accept()
            try:
                serve(conn, model, data)
            except (json.JSONDecodeError, UnicodeDecodeError, KeyError) as error:
                # Not the Copper bridge (e.g. a local HTTP port probe): drop it.
                print(f'ignoring non-protocol client: {error}', flush=True)
            except ConnectionError:
                print('bridge disconnected', flush=True)


def serve(conn, model, data):
    with conn, conn.makefile('rb') as reader:
        n = model.nu
        send_state(conn, model, data)
        for line in reader:
            message = json.loads(line)
            q = np.asarray(message['q'], dtype=float)
            qd = np.asarray(message.get('qd', [0] * n), dtype=float)
            problem = None
            if message.get('n', n) != n or q.shape != (n,) or qd.shape != (n,):
                problem = (f'joint count mismatch: sidecar serves {n} joints, '
                           f'command has n={message.get("n")}, '
                           f'len(q)={q.size}, len(qd)={qd.size}')
            elif not np.isfinite(q).all() or not np.isfinite(qd).all():
                problem = 'invalid joint command: non-finite value'
            if problem is not None:
                # Keep the request/response lockstep: answer with an error line.
                print(problem, flush=True)
                conn.sendall((json.dumps({'type': 'error', 'message': problem}) + '\n').encode())
                continue
            for _ in range(10):
                mujoco.mj_forward(model, data)
                data.ctrl[:] = np.clip(80 * (q - data.qpos[:n]) + 10 * (qd - data.qvel[:n]) + data.qfrc_bias[:n], -60, 60)
                mujoco.mj_step(model, data)
            send_state(conn, model, data)

if __name__ == '__main__':
    main()
