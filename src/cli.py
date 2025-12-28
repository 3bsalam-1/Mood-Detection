"""Simple CLI for training and prediction."""
from __future__ import annotations

import argparse
import sys

from src.train import build_model, load_dataset, train as train_model
from src.inference import MoodPredictor


def main(argv=None):
    parser = argparse.ArgumentParser(prog='mood', description='Mood Detection utilities')
    sub = parser.add_subparsers(dest='command')

    t = sub.add_parser('train', help='Train the model')
    t.add_argument('--epochs', type=int, default=50)

    p = sub.add_parser('predict', help='Predict from image file')
    p.add_argument('image', type=str)

    args = parser.parse_args(argv)

    if args.command == 'train':
        model = build_model()
        train_data, val_data, test_data = load_dataset()
        train_model(model, train_data, val_data, epochs=args.epochs)
    elif args.command == 'predict':
        predictor = MoodPredictor()
        res = predictor.predict_from_file(args.image)
        print(res)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
