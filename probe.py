#!/usr/bin/env python3
"""probe.py [substring ...] [--answers] [--list]   -- questions/answers per image."""
import json, os, sys
ROOT = os.path.dirname(os.path.abspath(__file__))
K = json.load(open(os.path.join(ROOT, 'ANSWER_KEY.json')))['images']
args = [a for a in sys.argv[1:] if not a.startswith('--')]
show = '--answers' in sys.argv or '-a' in sys.argv
if '--list' in sys.argv:
    print(f"{'file':58s} {'size':>10s} {'chars':>7s} {'cl-tok':>7s} {'ch/tok':>7s}")
    for k, v in K.items():
        print(f"{k:58s} {str(v['w'])+'x'+str(v['h']):>10s} {v['chars']:7d} "
              f"{v['claude_tokens']:7d} {v['chars_per_claude_token']:7.2f}")
    sys.exit()
hits = [(k, v) for k, v in K.items() if not args or any(a.lower() in k.lower() for a in args)]
if not hits: sys.exit("no match")
for k, v in hits:
    print(f"\n=== {k}  ({v['w']}x{v['h']}, {v['font']}, {v['chars']:,} chars, "
          f"{v['chars_per_claude_token']:.1f} ch/claude-token)")
    for p in v['probes']:
        print(f"  Q[{p['id']}] {p['q']}")
        if show:
            a = str(p['a'])
            print(f"    A: {a if len(a) < 400 else a[:400] + ' ...'}")
    if show: print(f"  full text: {v['groundtruth']}")
