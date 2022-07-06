import socket
import struct
import subprocess
import tempfile
from contextlib import contextmanager

import numpy as np
import pymesh

from .pcdexport import PCDExporter
from . import TUMColors, TUMBlue


class Matthew:
    def __init__(self, async=False, fullscreen=False, additional_data_folder=None, background_color=None):
        self._async = async
        self._command = ['matthew']
        if fullscreen:
            self._command.append('--fs')
        if additional_data_folder:
            self._command.append('-a')
            self._command.append(additional_data_folder)
        if background_color:
            self._command.append('--background')
            for c in background_color:
                self._command.append(str(c))

    def show_mesh(self, mesh):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.msh', delete=not self._async) as f:
            if 'color' not in mesh.get_attribute_names():
                mesh.add_attribute('color')
                c = np.empty_like(mesh.vertices)
                c[:, :] = TUMColors.hextofloats(TUMBlue)
                mesh.set_attribute('color', c)
            pymesh.save_mesh(f.name, mesh, *mesh.get_attribute_names())
            command = self._command + ['--file', f.name]
            print(f"Meshfile written to {f.name}.")
            (subprocess.Popen if self._async else subprocess.run)(command)

    def show_pointcloud(self, points, colors=None):
        exporter = PCDExporter(points)
        if colors is not None:
            exporter.add_color(colors)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pcd', delete=not self._async) as f:
            exporter.write_to(f)
            f.flush()
            print(f"Written to {f}.")

            command = self._command + ['--file', f.name, '--background', '0', '0', '0']
            if self._async:
                subprocess.Popen(command)
            else:
                subprocess.run(command)


@contextmanager
def matthew_point_steamer(ip="127.0.0.1", port=2222, offset=np.array([0, 0, 0]), interval=None):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def streamer(q):
        q_prime = q + offset
        if interval is not None:
            q_prime[q_prime < interval[0]] += 2*np.pi
            q_prime[q_prime > interval[1]] -= 2*np.pi
        sock.sendto(struct.pack("ddd", *q_prime), (ip, port))

    yield streamer
    sock.close()


