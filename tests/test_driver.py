import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from specdec.driver import running_scenario


SERVER = """
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert request["return_token_ids"] and request["temperature"] == 0
        result = {"prompt_token_ids": [1], "choices": [{"token_ids": [2]}]}
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps(result).encode())

print(os.getpid(), flush=True)
HTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
"""


class DriverTest(unittest.TestCase):
    def test_driver_reaps_only_its_server(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "server.py"
            script.write_text(SERVER)
            log = root / "server.log"
            sibling = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)"])
            try:
                def command(server, scenario):
                    return [sys.executable, str(script), str(server["port"])]

                with patch("specdec.driver.build_command", side_effect=command):
                    with running_scenario({"model": "fixture"}, {}, log, startup_timeout=5) as driver:
                        reply = driver.chat([{"role": "user", "content": "hello"}], 1)
                        self.assertEqual(reply["prompt_token_ids"], [1])
                        self.assertEqual(reply["token_ids"], [2])
                        pid = int(log.read_text().splitlines()[0])
                with self.assertRaises(ProcessLookupError):
                    os.kill(pid, 0)
                self.assertIsNone(sibling.poll())
            finally:
                sibling.terminate()
                sibling.wait()
