#!/usr/bin/env python3
"""Keren Visualizer backend server.

Serves static files and provides an API endpoint to run keren-sim.

Usage:
  python3 server.py [--port 8080] [--keren-sim /path/to/keren-sim]
"""

import argparse
import http.server
import json
import os
import subprocess
import tempfile
import urllib.parse

KEREN_SIM = None  # set by args

class VisualizerHandler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path == '/api/simulate':
            self._handle_simulate()
        elif parsed.path == '/api/simulate-trace':
            self._handle_simulate_trace()
        else:
            self.send_error(404, 'Not found')

    def _handle_simulate(self):
        """Run keren-sim on posted MLIR text, return results as JSON."""
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length)

        try:
            req = json.loads(body)
        except json.JSONDecodeError:
            self._json_response(400, {'error': 'Invalid JSON'})
            return

        mlir_text = req.get('mlir', '')
        entry = req.get('entry', 'main')
        inputs = req.get('inputs', [])  # list of JSON arrays

        if not mlir_text.strip():
            self._json_response(400, {'error': 'Empty MLIR input'})
            return

        # Write MLIR to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.mlir', delete=False) as f:
            f.write(mlir_text)
            tmp_path = f.name

        # Write inputs to a temp file if provided (avoids arg list too long)
        input_file_path = None
        if inputs:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as inf:
                json.dump(inputs, inf)
                input_file_path = inf.name

        try:
            cmd = [KEREN_SIM, '--json', f'--entry={entry}']
            if input_file_path:
                cmd.append(f'--input-file={input_file_path}')
            cmd.append(tmp_path)
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60
            )

            if result.returncode != 0:
                self._json_response(200, {
                    'success': False,
                    'error': result.stderr.strip() or 'keren-sim exited with error',
                    'stdout': result.stdout,
                })
            else:
                # Parse keren-sim JSON output
                try:
                    sim_output = json.loads(result.stdout)
                except json.JSONDecodeError:
                    sim_output = {'raw': result.stdout}

                self._json_response(200, {
                    'success': True,
                    'results': sim_output,
                    'stderr': result.stderr,
                })
        except subprocess.TimeoutExpired:
            self._json_response(200, {
                'success': False,
                'error': 'keren-sim timed out (60s)',
            })
        except FileNotFoundError:
            self._json_response(200, {
                'success': False,
                'error': f'keren-sim not found at: {KEREN_SIM}',
            })
        finally:
            os.unlink(tmp_path)
            if input_file_path:
                os.unlink(input_file_path)

    def _handle_simulate_trace(self):
        """Run keren-sim with --trace, return per-op values as JSON."""
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length)

        try:
            req = json.loads(body)
        except json.JSONDecodeError:
            self._json_response(400, {'error': 'Invalid JSON'})
            return

        mlir_text = req.get('mlir', '')
        entry = req.get('entry', 'main')
        inputs = req.get('inputs', [])

        if not mlir_text.strip():
            self._json_response(400, {'error': 'Empty MLIR input'})
            return

        with tempfile.NamedTemporaryFile(mode='w', suffix='.mlir', delete=False) as f:
            f.write(mlir_text)
            tmp_path = f.name

        trace_dir = tempfile.mkdtemp(prefix='keren-trace-')

        input_file_path = None
        if inputs:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as inf:
                json.dump(inputs, inf)
                input_file_path = inf.name

        try:
            cmd = [KEREN_SIM, '--json', f'--entry={entry}', f'--trace={trace_dir}']
            if input_file_path:
                cmd.append(f'--input-file={input_file_path}')
            cmd.append(tmp_path)
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60
            )

            # Parse results
            sim_results = {}
            if result.returncode == 0:
                try:
                    sim_results = json.loads(result.stdout)
                except json.JSONDecodeError:
                    sim_results = {'raw': result.stdout}

            # Parse trace index.csv
            trace_values = {}
            index_path = os.path.join(trace_dir, 'index.csv')
            if os.path.exists(index_path):
                import csv
                import struct
                with open(index_path) as cf:
                    reader = csv.reader(cf)
                    for row in reader:
                        if len(row) >= 3:
                            probe_id = row[0].strip()
                            dtype = row[1].strip()
                            npy_path = row[2].strip()
                            try:
                                value = _read_npy(npy_path)
                                trace_values[probe_id] = {'type': dtype, 'value': value}
                            except Exception as e:
                                trace_values[probe_id] = {'type': dtype, 'error': str(e)}

            self._json_response(200, {
                'success': result.returncode == 0,
                'results': sim_results,
                'trace': trace_values,
                'error': result.stderr.strip() if result.returncode != 0 else '',
            })

        except subprocess.TimeoutExpired:
            self._json_response(200, {'success': False, 'error': 'keren-sim timed out (60s)'})
        except FileNotFoundError:
            self._json_response(200, {'success': False, 'error': f'keren-sim not found at: {KEREN_SIM}'})
        finally:
            os.unlink(tmp_path)
            if input_file_path:
                os.unlink(input_file_path)
            import shutil
            shutil.rmtree(trace_dir, ignore_errors=True)

    def _json_response(self, code, data):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(body))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def log_message(self, format, *args):
        # Quieter logging
        if '/api/' in str(args[0]):
            super().log_message(format, *args)


def _read_npy(path):
    """Read a NumPy .npy file and return as nested list."""
    import struct

    with open(path, 'rb') as f:
        magic = f.read(6)
        if magic[:6] != b'\x93NUMPY':
            raise ValueError(f'Not a NumPy file: {path}')

        major = struct.unpack('B', f.read(1))[0]
        f.read(1)  # minor

        if major == 1:
            header_len = struct.unpack('<H', f.read(2))[0]
        else:
            header_len = struct.unpack('<I', f.read(4))[0]

        header = f.read(header_len).decode('latin1').strip()
        header_dict = eval(header)
        descr = header_dict['descr']
        shape = header_dict['shape']

        dtype_map = {
            '<f4': ('f', 4), '>f4': ('f', 4),
            '<f8': ('d', 8), '>f8': ('d', 8),
            '<i4': ('i', 4), '>i4': ('i', 4),
            '<i8': ('q', 8), '>i8': ('q', 8),
            '<i2': ('h', 2), '>i2': ('h', 2),
            '<u4': ('I', 4), '>u4': ('I', 4),
            '|b1': ('?', 1), '|u1': ('B', 1), '|i1': ('b', 1),
        }

        if descr not in dtype_map:
            return f'<unsupported dtype: {descr}>'

        fmt_char, item_size = dtype_map[descr]
        byte_order = '>' if descr.startswith('>') else '<'

        total = 1
        for s in shape:
            total *= s

        data = []
        for _ in range(total):
            raw = f.read(item_size)
            val = struct.unpack(f'{byte_order}{fmt_char}', raw)[0]
            data.append(val)

        return _reshape(data, shape)


def _reshape(flat, shape):
    if len(shape) == 0:
        return flat[0] if flat else None
    if len(shape) == 1:
        return flat
    size = 1
    for s in shape[1:]:
        size *= s
    return [_reshape(flat[i * size:(i + 1) * size], shape[1:]) for i in range(shape[0])]


def main():
    global KEREN_SIM

    parser = argparse.ArgumentParser(description='Keren Visualizer Server')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--keren-sim', default=None,
                        help='Path to keren-sim binary')
    args = parser.parse_args()

    # Auto-detect keren-sim
    if args.keren_sim:
        KEREN_SIM = args.keren_sim
    else:
        # Look relative to this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(script_dir, '..', 'build', 'tools', 'keren-sim'),
            os.path.join(script_dir, '..', 'build', 'keren-sim'),
        ]
        for c in candidates:
            if os.path.isfile(c) and os.access(c, os.X_OK):
                KEREN_SIM = os.path.abspath(c)
                break
        if not KEREN_SIM:
            # fallback: hope it's on PATH
            KEREN_SIM = 'keren-sim'

    print(f'Using keren-sim: {KEREN_SIM}')
    print(f'Serving on http://localhost:{args.port}')

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    server = http.server.HTTPServer(('', args.port), VisualizerHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nShutting down.')
        server.shutdown()


if __name__ == '__main__':
    main()
