"""Static check: every production function TIST calls must exist with a compatible arity."""
import ast, sys, pathlib

AUDIT = pathlib.Path('.')
def sigs(path):
    t = ast.parse(pathlib.Path(path).read_text(encoding='utf-8'))
    out = {}
    for n in ast.walk(t):
        if isinstance(n, ast.FunctionDef):
            a = n.args
            pos = [x.arg for x in a.posonlyargs + a.args]
            ndef = len(a.defaults)
            out[n.name] = (pos, len(pos) - ndef, bool(a.vararg))
    return out

prod = {}
for f in ['GPU_CPU/osm_behavioral.py','GPU_CPU/load_osm.py','GPU_CPU/utils_attention.py']:
    prod.update(sigs(f))

# (function, positional args passed, keyword names passed)
CALLS = [
    ('evaluate_osm_model', 4, {'run_id','sample_index'}),
    ('load_model', 1, set()),
    ('unload_model', 1, set()),
    ('_ensure_hooked_transformer', 2, set()),
    ('_ensure_nnsight_model', 2, set()),
    ('_nnsight_layer_proxies', 2, set()),
    ('_get_token_position', 3, set()),
]
bad = 0
for name, npos, kw in CALLS:
    if name not in prod:
        print(f'MISSING  {name}'); bad += 1; continue
    params, nreq, _ = prod[name]
    supplied = npos + len(kw)
    unknown = kw - set(params)
    if supplied < nreq or npos > len(params) or unknown:
        print(f'MISMATCH {name}: params={params} required={nreq} '
              f'passed_pos={npos} kw={sorted(kw)} unknown={sorted(unknown)}')
        bad += 1
    else:
        print(f'ok       {name}({npos} pos + {sorted(kw)}) vs {params}')
sys.exit(1 if bad else 0)
