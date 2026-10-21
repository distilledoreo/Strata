"""Opt-in raw token interface; no chat template, tokenization, retrieval or output parsing."""
import threading
import time


def validate(req, context, vocab=None):
    if not isinstance(req, dict):
        raise ValueError('selected prefill requires an object')
    allowed = {'token_ids', 'position_ids', 'original_prompt_length', 'max_tokens', 'sampling', 'model'}
    if set(req) - allowed:
        raise ValueError('unsupported selected prefill fields: ' + ', '.join(sorted(set(req) - allowed)))
    ids = req.get('token_ids')
    pos = req.get('position_ids')
    integer = lambda v: isinstance(v, int) and not isinstance(v, bool)
    if not isinstance(ids, list) or not ids or not all(integer(t) and t >= 0 for t in ids):
        raise ValueError('token_ids must be a nonempty array of nonnegative integers')
    if vocab and any(t >= vocab for t in ids):
        raise ValueError('a token id is outside the vocabulary')
    implicit = 'position_ids' not in req and 'original_prompt_length' not in req
    if implicit:
        pos = list(range(len(ids)))
    if not isinstance(pos, list) or len(pos) != len(ids) or not all(integer(p) for p in pos):
        raise ValueError('position_ids must contain one integer per token')
    original = len(ids) if implicit else req.get('original_prompt_length')
    maximum = req.get('max_tokens')
    if not integer(original) or original < len(ids) or original > context:
        raise ValueError('original_prompt_length must fit context and be at least the selected token count')
    if any(p < 0 or p >= original or (i and p <= pos[i-1]) for i, p in enumerate(pos)):
        raise ValueError('position_ids must be strictly increasing within original_prompt_length')
    if not integer(maximum) or maximum < 1 or maximum > context - original - 8:
        raise ValueError('original prompt plus max_tokens and safety margin must fit context')
    sampling = req.get('sampling', {})
    permitted = {'temperature', 'top_p', 'top_k', 'min_p', 'seed', 'penalty_last_n',
                 'penalty_repeat', 'penalty_freq', 'penalty_present', 'logit_bias'}
    if not isinstance(sampling, dict) or set(sampling) - permitted:
        raise ValueError('unsupported selected prefill sampling fields')
    return list(ids), list(pos), original, maximum, dict(sampling)


def generate(service, req, cancel=None):
    """One serialized native request. Ordinary endpoints retain their existing behavior."""
    from serve.server import check_request_sampling, bias_api, EngineDied
    engine = service.engine
    args = getattr(engine, 'spawn', (None, []))[1]
    if '--selected-prefill' not in args or getattr(engine, 'batch', 0):
        raise ValueError('start the experimental engine with --selected-prefill, without batching')
    ids, pos, original, maximum, sampling = validate(
        req, engine.max_context or getattr(engine, 'known_ctx', 0),
        len(getattr(service.tok, 'tokens', ())) or None)
    if req.get('model') not in (None, service.model):
        raise ValueError('model does not match this server')
    check_request_sampling(sampling)
    bias_api.validate_request(sampling, engine, len(getattr(service.tok, 'tokens', ())) or None)
    explicit = 'position_ids' in req or 'original_prompt_length' in req
    if explicit:
        sampling['_selected_prefill'] = {'position_ids': pos, 'original_prompt_length': original}
    cancel = cancel or threading.Event()
    started = time.perf_counter()
    first = None
    output = []
    service.last_request_at = time.time()
    with service.fifo:
        service.ensure_loaded()
        if engine.info.get('selected_prefill') != 1:
            raise ValueError('engine does not advertise selected-prefill support; rebuild it first')
        # Recheck after a restart against the actual loaded context.
        validate(req, engine.max_context, len(getattr(service.tok, 'tokens', ())) or None)
        try:
            for token in engine.generate(ids, maximum, sampling, cancel):
                if token is not None:
                    if first is None:
                        first = time.perf_counter()
                    output.append(token)
        finally:
            service.last_request_at = time.time()
        audit = engine.position_audit
        expected = {'original_prompt_length': original, 'position_ids': pos, 'next_position': original}
        if explicit and audit != expected:
            raise EngineDied('engine did not confirm the requested device position table')
        timings = dict(engine.last)
        if not explicit:
            audit = expected
    return {'model': service.model, 'token_ids': output,
            'prefill': {'token_ids': ids, **audit, 'storage': 'compact', 'device_table_verified': explicit,
                        'mode': 'original_positions' if explicit else 'implicit_contiguous'},
            'usage': {'prompt_tokens': len(ids), 'original_prompt_tokens': original,
                      'completion_tokens': len(output)},
            'timings': {'first_token_monotonic_s': first,
                        'ttft_s': None if first is None else first-started,
                        'total_s': time.perf_counter()-started, 'engine': timings}}
