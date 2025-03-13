import argparse
import os
from custom.clip_ebc_onnx import ClipEBCOnnx

def parse_args():
    parser = argparse.ArgumentParser(description='CLIP-EBC Crowd Counting (ONNX)')
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--model', default='clip_ebc_model.onnx', help='Path to ONNX model')
    parser.add_argument('--visualize', choices=['density', 'dots', 'all', 'none'], 
                        default='none', help='Visualization type')
    parser.add_argument('--save', action='store_true', 
                        help='Save visualization results')
    parser.add_argument('--output-dir', default='results', 
                        help='Directory to save results')
    
    # 시각화 관련 매개변수
    parser.add_argument('--alpha', type=float, default=0.5, 
                        help='Alpha value for density map')
    parser.add_argument('--dot-size', type=int, default=20, 
                        help='Dot size for dot visualization')
    parser.add_argument('--sigma', type=float, default=1, 
                        help='Sigma value for Gaussian filter')
    parser.add_argument('--percentile', type=float, default=97, 
                        help='Percentile threshold for dot visualization')
    

    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 모델 초기화 - ONNX 버전
    model = ClipEBCOnnx(
        onnx_model_path=args.model
    )
    
    # 출력 디렉토리 생성
    if args.save:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # 예측 수행
    count = model.predict(args.image)
    print(f"예측된 군중 수: {count:.2f}")
    
    # 시각화
    if args.visualize in ['density', 'all']:
        save_path = os.path.join(args.output_dir, 'density_map.png') if args.save else None
        fig, density_map = model.visualize_density_map(
            alpha=args.alpha,
            save=args.save,
            save_path=save_path
        )
    
    if args.visualize in ['dots', 'all']:
        save_path = os.path.join(args.output_dir, 'dot_map.png') if args.save else None
        canvas, dot_map = model.visualize_dots(
            dot_size=args.dot_size,
            sigma=args.sigma,
            percentile=args.percentile,
            save=args.save,
            save_path=save_path
        )
        
        # matplotlib figure 닫기 (메모리 누수 방지)
        if args.visualize in ['density', 'all']:
            import matplotlib.pyplot as plt
            plt.close(fig)
        
        if args.visualize in ['dots', 'all']:
            import matplotlib.pyplot as plt
            plt.close(canvas.figure)

if __name__ == "__main__":
    main()