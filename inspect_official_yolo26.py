"""Read official YOLO26 P3 layer shape from an actual FOUR-CLASS trained checkpoint."""
import argparse
import torch
from rotation_detector.official_model import discover_official_p3, check_official_yolo26_graph


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model', required=True, help='4-class best.pt from train_official_baseline.py')
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--device', default='cpu')
    args=p.parse_args()
    from ultralytics import YOLO
    net=YOLO(args.model).model.to(torch.device(args.device))
    check_official_yolo26_graph(net)
    spec=discover_official_p3(net, args.imgsz)
    print(f'Official YOLO26 P3: index={spec.layer}; C={spec.channels}; HxW={spec.height}x{spec.width}; nc={spec.num_classes}')

if __name__ == '__main__': main()
