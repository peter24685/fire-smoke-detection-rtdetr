"""RT-DETR fire and smoke detection: a portable CLI for training, validation, and inference."""
import argparse
import json
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('prepare', help='Generate a local dataset configuration without changing labels')
    prepare.add_argument('--root', type=Path, required=True)
    prepare.add_argument('--output', type=Path, default=Path('configs/local.yaml'))
    for name in ('train', 'val', 'predict'):
        p = sub.add_parser(name)
        p.add_argument('--weights', required=True)
        p.add_argument('--device', default='cpu')
        p.add_argument('--imgsz', type=int, default=640)
        p.add_argument('--project', default='runs')
        p.add_argument('--name', default=name)
        if name == 'predict':
            p.add_argument('--source', required=True)
            p.add_argument('--conf', type=float, default=.25)
        else:
            p.add_argument('--data', type=Path, default=Path('configs/local.yaml'))
            p.add_argument('--batch', type=int, default=2)
            p.add_argument('--workers', type=int, default=0)
            if name == 'train':
                p.add_argument('--epochs', type=int, default=1)
                p.add_argument('--seed', type=int, default=0)
            else:
                p.add_argument('--split', choices=('val', 'test'), default='val')
    return parser


def prepare(root, output):
    root = root.resolve()
    missing = [str(root / s / k) for s in ('train', 'val', 'test')
               for k in ('images', 'labels') if not (root / s / k).is_dir()]
    if missing:
        raise ValueError('Missing dataset directories: ' + ', '.join(missing))
    output.parent.mkdir(parents=True, exist_ok=True)
    # JSON strings are valid YAML strings and safely represent Windows paths with spaces.
    output.write_text('path: ' + json.dumps(root.as_posix(), ensure_ascii=False)
                      + '\ntrain: train/images\nval: val/images\ntest: test/images\n'
                        'names:\n  0: smoke\n  1: fire\n', encoding='utf-8')
    return output


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.command == 'prepare':
        try:
            print(prepare(args.root, args.output))
        except ValueError as error:
            parser.error(str(error))
        return
    if args.imgsz <= 0 or (hasattr(args, 'batch') and args.batch <= 0):
        parser.error('imgsz and batch must be positive')
    if args.command == 'train' and args.epochs <= 0:
        parser.error('epochs must be positive')
    if args.command == 'predict' and not 0 <= args.conf <= 1:
        parser.error('conf must be between 0 and 1')
    if args.command != 'predict' and not args.data.is_file():
        parser.error('Dataset configuration not found; run prepare first')
    # Only training may download pretrained weights; inference requires a local trained checkpoint.
    if args.command != 'train' and not Path(args.weights).is_file():
        parser.error('Trained weights not found; provide a local best.pt path')
    if args.command == 'predict' and not Path(args.source).exists():
        parser.error('Input image or directory not found')
    from ultralytics import RTDETR
    model = RTDETR(args.weights)
    common = dict(imgsz=args.imgsz, device=args.device,
                  project=str(Path(args.project).resolve()), name=args.name)
    if args.command == 'train':
        model.train(data=str(args.data.resolve()), epochs=args.epochs, batch=args.batch,
                    workers=args.workers, seed=args.seed, **common)
    elif args.command == 'val':
        result = model.val(data=str(args.data.resolve()), split=args.split,
                           batch=args.batch, workers=args.workers, **common)
        print(json.dumps(result.results_dict, ensure_ascii=False, indent=2))
    else:
        results = model.predict(source=args.source, conf=args.conf, save=True,
                                stream=True, **common)
        for result in results:
            print(json.dumps({'image': Path(result.path).name,
                              'detections': len(result.boxes),
                              'save_dir': str(result.save_dir)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
