import json, sys
V = '/root/autodl-tmp/project/code-video-temporal'
REF = '/root/autodl-tmp/project/code/outputs/pv_baseline'
A = f'{V}/outputs/pv_gate_wired'
B = f'{V}/outputs/pv_mixed_v1'

def load_metrics(p): return json.load(open(p + '/metrics.json'))
def load_recs(p): return json.load(open(p + '/metrics_records.json'))

fail, notes = [], []

# ---- 1) Gate: Arm A must reproduce the pv_baseline reference exactly ----
ref_m, a_m = load_metrics(REF), load_metrics(A)
if ref_m.get('tolerance') != a_m.get('tolerance'):
    fail.append(f"gate tolerance: {ref_m.get('tolerance')} vs {a_m.get('tolerance')}")
if ref_m['overall_score'] != a_m['overall_score']:
    fail.append(f"gate overall: {ref_m['overall_score']} vs {a_m['overall_score']}")
for task, v in ref_m['per_task'].items():
    if v != a_m['per_task'].get(task):
        fail.append(f"gate per_task {task}: {v} vs {a_m['per_task'].get(task)}")
print('[gate] A per_task:', json.dumps(a_m['per_task']))
print('[gate] A overall:', a_m['overall_score'], ' classification matched:',
      a_m.get('classification_matched'), '/', ref_m.get('classification_matched'))

# ---- 2) Routing assertion: image_seg/image_cls columns identical between arms ----
ra = {r['key']: r for r in load_recs(A)}
rb = {r['key']: r for r in load_recs(B)}
if set(ra) != set(rb):
    fail.append(f'route keys mismatch: {len(set(ra) ^ set(rb))} keys differ')
for task in ('image_seg', 'image_cls'):
    field = 'prob' if task == 'image_cls' else 'score'
    ka = {k: v for k, v in ra.items() if v['task'] == task}
    kb = {k: v for k, v in rb.items() if k in rb and rb[k]['task'] == task}
    exact = sum(1 for k in ka if kb.get(k) and ka[k] == kb[k])
    maxd = max((abs(float(ka[k][field]) - float(kb[k][field])) for k in ka if k in kb), default=0.0)
    print(f'[route] {task}: n={len(ka)} record_exact={exact} max_{field}_diff={maxd:.3e}')
    if exact == len(ka):
        notes.append(f'route {task}: EXACT ({len(ka)} entries)')
    elif maxd <= 1e-6:
        notes.append(f'route {task}: TOLERANCE_PASS (maxd={maxd:.2e})')
    else:
        fail.append(f'route {task}: exact={exact}/{len(ka)} max{field}d={maxd:.3e}')

# ---- 3) Mixed arm scores + direction vs gate arm ----
b_m = load_metrics(B)
print('[mixed] B per_task:', json.dumps(b_m['per_task']))
print('[mixed] B overall:', b_m['overall_score'])
print('[direction] overall A->B:', round(b_m['overall_score'] - a_m['overall_score'], 6))
for t in sorted(a_m['per_task']):
    d = b_m['per_task'].get(t)
    if d is not None:
        print(f'[direction] {t}: {round(d["score"] - a_m["per_task"][t]["score"], 6)}')

print('---')
for n in notes: print('NOTE:', n)
if fail:
    print('PHASE4_ASSERT_FAIL')
    for f in fail: print(' -', f)
    sys.exit(1)
print('PHASE4_ASSERT_PASS')
