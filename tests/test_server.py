"""The voice server, with stand-ins for speech and the model."""

import base64
import http.client
import json
import tempfile
import threading
import unittest
from pathlib import Path

from keel import cli, config, server
from tests.test_core import NOW, FakeLLM

SESSION = "test-session-0001"


class FakeSpeech:
    def __init__(self, heard="I skipped my run again today."):
        self.heard = heard

    def transcribe(self, audio, model):
        return self.heard

    def speak(self, text, voice):
        return b"RIFFfakeWAVE" + text.encode()


class ServerTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cfg = config.load(self._tmp.name, embed_model="none")
        self.speech = FakeSpeech()
        keel = server.Keel(self.cfg, self.speech, voice="af_heart", stt_model="x", now=NOW, llm=FakeLLM())
        self.httpd = server.make_server(keel, "127.0.0.1", 0)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self._tmp.cleanup()

    def request(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.request(method, path, body=body, headers=headers or {})
        resp = conn.getresponse()
        data = resp.read()
        conn.close()
        return resp.status, resp.getheader("Content-Type"), data

    def post(self, path, body, ctype, **headers):
        status, _, data = self.request("POST", path, body, {"Content-Type": ctype, "X-Keel": "1", **headers})
        return status, json.loads(data)

    def recordings(self):
        return sorted(p.name for p in self.cfg.audio_dir.glob("*")) if self.cfg.audio_dir.exists() else []

    def test_the_page(self):
        status, ctype, body = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertTrue(ctype.startswith("text/html"))
        self.assertIn(b"Hold to talk", body)

    def test_a_typed_turn_is_answered_out_loud(self):
        status, data = self.post(f"/api/text?session={SESSION}", json.dumps({"text": "Long day."}), "application/json")
        self.assertEqual(status, 200)
        self.assertEqual(data["you"], "Long day.")
        self.assertTrue(base64.b64decode(data["audio"]).startswith(b"RIFF"))

    def test_a_spoken_entry_keeps_its_recording_until_forgotten(self):
        status, data = self.post(f"/api/turn?session={SESSION}", b"\x1aE\xdf\xa3" + b"0" * 4000, "audio/webm;codecs=opus")
        self.assertEqual(status, 200)
        self.assertEqual(data["you"], "I skipped my run again today.")
        files = self.recordings()
        self.assertEqual(len(files), 1)
        self.assertTrue(files[0].endswith(".webm"))
        status, data = self.post(f"/api/text?session={SESSION}", json.dumps({"text": "forget that"}), "application/json")
        self.assertTrue(data["keel"].startswith("Gone."))
        self.assertEqual(self.recordings(), [])

    def test_questions_and_silence_keep_no_recording(self):
        self.speech.heard = "Am I avoiding the pricing decision?"
        self.post(f"/api/turn?session={SESSION}", b"0" * 4000, "audio/webm")
        self.assertEqual(self.recordings(), [])
        self.speech.heard = ""
        status, data = self.post(f"/api/turn?session={SESSION}", b"0" * 4000, "audio/webm")
        self.assertIn("didn't catch", data["error"])
        self.assertEqual(self.recordings(), [])

    def test_guards(self):
        body = json.dumps({"text": "hi"})
        status, _, _ = self.request("POST", f"/api/text?session={SESSION}", body, {"Content-Type": "application/json"})
        self.assertEqual(status, 403, "no X-Keel header")
        status, _, _ = self.request(
            "POST", f"/api/text?session={SESSION}", body, {"Content-Type": "application/json", "X-Keel": "1", "Host": "evil.example:8095"}
        )
        self.assertEqual(status, 403, "a page pointing its own domain at us")
        status, _, _ = self.request("GET", "/", headers={"Host": "evil.example"})
        self.assertEqual(status, 403)
        status, _, _ = self.request("OPTIONS", f"/api/text?session={SESSION}")
        self.assertEqual(status, 403, "no preflight is ever approved")
        status, _ = self.post("/api/text?session=x", body, "application/json")
        self.assertEqual(status, 400, "bad session id")

    def test_through_tailscale_it_has_to_be_you(self):
        self.httpd.RequestHandlerClass.allowed_hosts = self.httpd.RequestHandlerClass.allowed_hosts | {"box.tail.ts.net"}
        self.httpd.RequestHandlerClass.allowed_users = {"me@example.com"}
        via = {"X-Forwarded-For": "100.64.0.7", "X-Forwarded-Host": "box.tail.ts.net:8095"}
        status, _, _ = self.request("GET", "/", headers=via)
        self.assertEqual(status, 403, "forwarded with no Tailscale login")
        status, _, _ = self.request("GET", "/", headers={**via, "Tailscale-User-Login": "someone@else.com"})
        self.assertEqual(status, 403, "someone else on the tailnet")
        status, _, _ = self.request("GET", "/", headers={**via, "Tailscale-User-Login": "Me@Example.com"})
        self.assertEqual(status, 200)
        status, _, _ = self.request(
            "GET", "/", headers={**via, "X-Forwarded-Host": "evil.example", "Tailscale-User-Login": "me@example.com"}
        )
        self.assertEqual(status, 403, "forwarded for a name we don't answer to")

    def test_it_only_listens_on_this_machine(self):
        self.assertEqual(cli.main(["--home", self._tmp.name, "serve", "--host", "0.0.0.0"]), 2)


if __name__ == "__main__":
    unittest.main()
