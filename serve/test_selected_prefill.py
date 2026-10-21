"""Run: python -m unittest serve.test_selected_prefill -v (no model/GPU)."""
import contextlib
import io
import json
import queue
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from serve.selected_prefill import generate, validate
from serve.server import ByteTokenizer, EngineDied, MockEngine, Service, StrataEngine, serve
from serve.frontend import ChatTemplate


class SelectedPrefill(unittest.TestCase):
    def request(self, **kw):
        return dict(token_ids=[10, 20, 30], position_ids=[0, 2, 4],
                    original_prompt_length=7, max_tokens=3, **kw)

    def test_sparse_and_full(self):
        self.assertEqual(validate(self.request(), 100)[1:4], ([0, 2, 4], 7, 3))
        r = self.request(); r['position_ids'] = [0, 1, 2]; r['original_prompt_length'] = 3
        self.assertEqual(validate(r, 100)[1:4], ([0, 1, 2], 3, 3))

    def test_reject_malformed_or_out_of_range(self):
        mutations = [('token_ids', []), ('token_ids', [True, 2, 3]), ('token_ids', [-1, 2, 3]),
                     ('token_ids', [1, 2, 1000]), ('position_ids', [0, 2]),
                     ('position_ids', [0, 0, 2]), ('position_ids', [2, 1, 3]),
                     ('position_ids', [-1, 2, 4]), ('position_ids', [0, 2, 7]),
                     ('position_ids', [0, 2, False]), ('original_prompt_length', 2),
                     ('original_prompt_length', True), ('original_prompt_length', 100),
                     ('max_tokens', 0), ('max_tokens', True), ('max_tokens', 86),
                     ('sampling', {'strata_prefix': {}}), ('stream', True)]
        for key, value in mutations:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                r = self.request(); r[key] = value
                validate(r, 100, vocab=100)

    def engine(self):
        e = object.__new__(StrataEngine)
        e.proc = mock.Mock(); e.proc.stdin = io.StringIO(); e.proc.poll.return_value = None
        e.lines = queue.Queue(); e.can_stop = False; e.max_context = 100
        e.silence_s = 0; e.info = {'selected_prefill': 1}
        e.lines.put('POSITION_IDS 7 0,2,4 7')
        e.lines.put('T 55'); e.lines.put('DONE 1 3 1.0 2.0 length')
        return e

    def test_wire_preserves_ids_and_positions(self):
        e = self.engine()
        out = list(e.generate([10, 20, 30], 3, {'_selected_prefill': {
            'position_ids': [0, 2, 4], 'original_prompt_length': 7}}, threading.Event()))
        self.assertEqual(out, [55])
        self.assertEqual(e.proc.stdin.getvalue(), 'GEN 3 positions=0,2,4 original_length=7 10,20,30\n')
        self.assertEqual(e.position_audit['next_position'], 7)

    def test_ordinary_wire_unchanged(self):
        e = self.engine()
        list(e.generate([10, 20, 30], 3, {}, threading.Event()))
        self.assertEqual(e.proc.stdin.getvalue(), 'GEN 3 10,20,30\n')

    def service(self):
        e = self.engine(); e.spawn = (None, ['--selected-prefill'])
        return SimpleNamespace(engine=e, model='test', tok=SimpleNamespace(tokens=range(100)),
                               fifo=threading.Lock(), ensure_loaded=lambda: None)

    def test_raw_output_and_audit(self):
        svc = self.service()
        result = generate(svc, self.request())
        self.assertEqual(result['token_ids'], [55])
        self.assertEqual(result['prefill']['token_ids'], [10, 20, 30])
        self.assertTrue(result['prefill']['device_table_verified'])
        self.assertEqual(result['usage']['prompt_tokens'], 3)

    def test_missing_or_wrong_ack_fails_closed(self):
        for ack in (None, 'POSITION_IDS 6 0,2,4 6'):
            svc = self.service(); svc.engine.lines = queue.Queue()
            if ack: svc.engine.lines.put(ack)
            svc.engine.lines.put('T 55'); svc.engine.lines.put('DONE 1 3 1.0 2.0 length')
            with self.subTest(ack=ack), self.assertRaises(EngineDied):
                generate(svc, self.request())

    def test_implicit_contiguous_control(self):
        svc = self.service()
        result = generate(svc, {'token_ids': [10, 20, 30], 'max_tokens': 3})
        self.assertEqual(svc.engine.proc.stdin.getvalue(), 'GEN 3 10,20,30\n')
        self.assertEqual(result['prefill']['mode'], 'implicit_contiguous')
        self.assertFalse(result['prefill']['device_table_verified'])

    def test_engine_capability_required(self):
        svc = self.service(); svc.engine.info = {}
        with self.assertRaises(ValueError): generate(svc, self.request())
        svc.engine.spawn = (None, [])
        with self.assertRaises(ValueError): generate(svc, self.request())

    def test_http_route(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, 'unused', max_context=100), tok,
                      ChatTemplate(Path(__file__).parent / 'chat_template.jinja'))
        svc.engine = self.service().engine
        svc.ensure_loaded = lambda: None
        httpd = serve(svc, port=0)
        try:
            url = f'http://127.0.0.1:{httpd.server_address[1]}/v1/strata/generate'
            def post(body):
                return urllib.request.urlopen(urllib.request.Request(url,
                    data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'}), timeout=5)
            with contextlib.closing(post(self.request(model=svc.model))) as res:
                self.assertEqual(json.load(res)['token_ids'], [55])
            bad = self.request(); bad['position_ids'] = [0, 0, 4]
            with self.assertRaises(urllib.error.HTTPError) as err: post(bad)
            self.assertEqual(err.exception.code, 400)
        finally:
            httpd.shutdown(); httpd.server_close()


if __name__ == '__main__':
    unittest.main()
