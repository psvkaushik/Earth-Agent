#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare end-to-end accuracy between a tool-enabled run and its no-tools ablation
counterpart for the same model. Both inputs are the JSON files produced by
evaluate/end_to_end_single.py (run it once per directory before using this).
"""
import json
import argparse


def load_summary(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)['summary']


def main():
    parser = argparse.ArgumentParser(description="Compare tool-enabled vs no-tools ablation accuracy")
    parser.add_argument('--with-tools', required=True,
                         help="Path to end_to_end_evaluation_results.json for the tool-enabled run")
    parser.add_argument('--no-tools', required=True,
                         help="Path to end_to_end_evaluation_results.json for the no-tools ablation run")
    parser.add_argument('--label', default='model', help="Label for this comparison (e.g. model name)")
    args = parser.parse_args()

    with_tools = load_summary(args.with_tools)
    no_tools = load_summary(args.no_tools)

    delta = no_tools['accuracy_rate'] - with_tools['accuracy_rate']

    print("=" * 60)
    print(f"Ablation comparison: {args.label}")
    print("=" * 60)
    print(f"{'Metric':<20} {'With tools':<15} {'No tools':<15} {'Delta':<10}")
    print("-" * 60)
    print(f"{'Accuracy':<20} {with_tools['accuracy_rate']*100:<14.2f}% {no_tools['accuracy_rate']*100:<14.2f}% {delta*100:+.2f}%")
    print(f"{'Fail rate':<20} {with_tools['fail_rate']*100:<14.2f}% {no_tools['fail_rate']*100:<14.2f}% "
          f"{(no_tools['fail_rate']-with_tools['fail_rate'])*100:+.2f}%")
    print("=" * 60)
    print("Note: 'average_efficiency' is not meaningful for the no-tools run (no tool calls by construction).")


if __name__ == "__main__":
    main()
