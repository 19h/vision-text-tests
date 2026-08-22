#!/usr/bin/env python3
"""Approximate sample size for a paired binary equivalence study."""
import argparse
import math
from statistics import NormalDist


def required_n(margin, discordance, alpha=.05, power=.80, expected_difference=0.0):
    if abs(expected_difference) >= margin:
        raise ValueError('expected difference must lie inside the equivalence margin')
    z_alpha = NormalDist().inv_cdf(1 - alpha)
    z_power = NormalDist().inv_cdf(power)
    effective_margin = margin - abs(expected_difference)
    variance = max(1e-12, discordance - expected_difference ** 2)
    return math.ceil((z_alpha + z_power) ** 2 * variance / effective_margin ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--margin', type=float, default=.10)
    ap.add_argument('--discordance', type=float, nargs='+', default=[.10, .20, .30, .50])
    ap.add_argument('--alpha', type=float, default=.05,
                    help='one-sided alpha for each TOST boundary')
    ap.add_argument('--power', type=float, default=.80)
    ap.add_argument('--expected-difference', type=float, default=0.0)
    a = ap.parse_args()
    print('paired binary equivalence (normal approximation; confirm by simulation before launch)')
    print(f'margin={a.margin:.3f}, alpha={a.alpha:.3f}, power={a.power:.2f}, '
          f'expected difference={a.expected_difference:+.3f}')
    print('discordance  required paired items')
    for d in a.discordance:
        print(f'{d:10.3f}  {required_n(a.margin, d, a.alpha, a.power, a.expected_difference):21d}')


if __name__ == '__main__':
    main()
