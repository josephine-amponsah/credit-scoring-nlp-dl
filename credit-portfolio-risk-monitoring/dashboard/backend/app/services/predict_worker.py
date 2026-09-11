"""Worker script to load a pickled model and predict probabilities for an input CSV.

This script is intended to be invoked as a subprocess to isolate native crashes
that can occur during unpickling or native library calls.

Usage:
  python predict_worker.py --model /abs/path/to/model.pkl --input /abs/path/to/input.csv --output /abs/path/to/out.json
"""
import argparse
import json
import sys
import os
import joblib
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()

    # load input
    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        print(json.dumps({'error': f'failed to read input: {e}'}))
        sys.exit(2)

    try:
        model = joblib.load(args.model)
    except Exception as e:
        print(json.dumps({'error': f'failed to load model: {e}'}))
        sys.exit(3)

    # select features
    try:
        if hasattr(model, 'feature_names_in_'):
            features = list(model.feature_names_in_)
        else:
            features = df.select_dtypes(include=['number']).columns.tolist()
        if not features:
            raise RuntimeError('no feature columns found')
        Xf = df[features].fillna(0)
    except Exception as e:
        print(json.dumps({'error': f'failed to prepare features: {e}'}))
        sys.exit(4)

    try:
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(Xf)
            if probs.ndim == 2 and probs.shape[1] > 1:
                out = probs[:, 1].tolist()
            else:
                out = probs.ravel().tolist()
        elif hasattr(model, 'predict'):
            out = model.predict(Xf).tolist()
        else:
            raise RuntimeError('model has no predict_proba or predict')
    except Exception as e:
        print(json.dumps({'error': f'prediction failed: {e}'}))
        sys.exit(5)

    try:
        with open(args.output, 'w') as f:
            json.dump({'probs': out}, f)
    except Exception as e:
        print(json.dumps({'error': f'failed to write output: {e}'}))
        sys.exit(6)


if __name__ == '__main__':
    main()
