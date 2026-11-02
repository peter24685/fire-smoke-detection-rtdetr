"""RT-DETR 火焰烟雾检测：可移植的训练、验证与预测入口。"""
import argparse
import json
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('prepare', help='生成本机数据配置，不修改标签')
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
        raise ValueError('缺少数据目录：' + ', '.join(missing))
    output.parent.mkdir(parents=True, exist_ok=True)
    # JSON 字符串也是合法 YAML 字符串，可安全保存含空格的 Windows 路径。
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
        parser.error('imgsz 和 batch 必须为正数')
    if args.command == 'train' and args.epochs <= 0:
        parser.error('epochs 必须为正数')
    if args.command == 'predict' and not 0 <= args.conf <= 1:
        parser.error('conf 必须在 0 到 1 之间')
    if args.command != 'predict' and not args.data.is_file():
        parser.error('找不到数据配置，请先运行 prepare')
    # 只允许训练入口使用框架下载预训练模型；预测需提供实际训练权重。
    if args.command != 'train' and not Path(args.weights).is_file():
        parser.error('找不到训练权重，请提供本地 best.pt 路径')
    if args.command == 'predict' and not Path(args.source).exists():
        parser.error('找不到输入图片或目录')
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
