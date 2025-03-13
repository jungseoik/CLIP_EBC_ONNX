import os
import sys
import torch
from torchvision.transforms.functional import normalize, to_pil_image
from torchvision.transforms import ToTensor, Normalize
import matplotlib.pyplot as plt
import json
from models import get_model
from utils import resize_density_map, sliding_window_predict
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans
import datetime
from typing import Optional
from typing import Union

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ClipEBC:
    """
    CLIP-EBC (Efficient Boundary Counting) 이미지 처리 클래스입니다.
    
    CLIP 모델을 사용하여 이미지를 처리하며, 슬라이딩 윈도우 예측 기능을 포함한
    다양한 설정 옵션을 제공합니다.
    
    Attributes:
        truncation (int): 잘라내기 매개변수. 기본값 4.
        reduction (int): 축소 비율. 기본값 8.
        granularity (str): 세분화 수준. 기본값 "fine".
        anchor_points (str): 앵커 포인트 방법. 기본값 "average".
        model_name (str): CLIP 모델 이름. 기본값 "clip_vit_b_16".
        input_size (int): 입력 이미지 크기. 기본값 224.
        window_size (int): 슬라이딩 윈도우 크기. 기본값 224.
        stride (int): 슬라이딩 윈도우 이동 간격. 기본값 224.
        prompt_type (str): 프롬프트 유형. 기본값 "word".
        dataset_name (str): 데이터셋 이름. 기본값 "qnrf".
        num_vpt (int): 비주얼 프롬프트 토큰 수. 기본값 32.
        vpt_drop (float): 비주얼 프롬프트 토큰 드롭아웃 비율. 기본값 0.0.
        deep_vpt (bool): 깊은 비주얼 프롬프트 토큰 사용 여부. 기본값 True.
        mean (tuple): 정규화를 위한 평균값. 기본값 (0.485, 0.456, 0.406).
        std (tuple): 정규화를 위한 표준편차값. 기본값 (0.229, 0.224, 0.225).
    """
    
    def __init__(self,
                 truncation=4,
                 reduction=8,
                 granularity="fine",
                 anchor_points="average",
                 model_name="clip_vit_b_16",
                 input_size=224,
                 window_size=224,
                 stride=224,
                 prompt_type="word",
                 dataset_name="qnrf",
                 num_vpt=32,
                 vpt_drop=0.,
                 deep_vpt=True,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225),
                 config_dir="configs"):
        """CLIPEBC 클래스를 설정 매개변수와 함께 초기화합니다."""
        self.truncation = truncation
        self.reduction = reduction
        self.granularity = granularity
        self.anchor_points_type = anchor_points  # 원래 입력값 저장
        self.model_name = model_name
        self.input_size = input_size
        self.window_size = window_size
        self.stride = stride
        self.prompt_type = prompt_type
        self.dataset_name = dataset_name
        self.num_vpt = num_vpt
        self.vpt_drop = vpt_drop
        self.deep_vpt = deep_vpt
        self.mean = mean
        self.std = std
        self.config_dir = config_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.bins = None
        self.anchor_points = None
        self.model = None
        
        # 초기 설정 로드 및 모델 초기화
        self._load_config()
        self._initialize_model()
        
    def _load_config(self):
        """설정 파일을 로드하고 bins와 anchor_points를 설정합니다."""
        config_path = os.path.join(self.config_dir, f"reduction_{self.reduction}.json")
        with open(config_path, "r") as f:
            config = json.load(f)[str(self.truncation)][self.dataset_name]
        
        self.bins = config["bins"][self.granularity]
        self.bins = [(float(b[0]), float(b[1])) for b in self.bins]
        
        if self.anchor_points_type == "average":
            self.anchor_points = config["anchor_points"][self.granularity]["average"]
        else:
            self.anchor_points = config["anchor_points"][self.granularity]["middle"]
        self.anchor_points = [float(p) for p in self.anchor_points]
        
    def _initialize_model(self):
        """CLIP 모델을 초기화합니다."""
        self.model = get_model(
            backbone=self.model_name,
            input_size=self.input_size,
            reduction=self.reduction,
            bins=self.bins,
            anchor_points=self.anchor_points,
            prompt_type=self.prompt_type,
            num_vpt=self.num_vpt,
            vpt_drop=self.vpt_drop,
            deep_vpt=self.deep_vpt
        )

        ckpt_path = "assets/CLIP_EBC_nwpu_rmse.pth"
        ckpt = torch.load(ckpt_path, map_location=device)
        self.model.load_state_dict(ckpt)
        self.model = self.model.to(device)
        self.model.eval()

    def visualize_density_map(self, alpha: float = 0.5, save: bool = False, 
                            save_path: Optional[str] = None):
        """
        현재 저장된 예측 결과를 시각화합니다.
        
        Args:
            alpha (float): density map의 투명도 (0~1). 기본값 0.5
            save (bool): 시각화 결과를 이미지로 저장할지 여부. 기본값 False
            save_path (str, optional): 저장할 경로. None일 경우 현재 디렉토리에 자동 생성된 이름으로 저장.
                기본값 None
                
        Returns:
            Tuple[matplotlib.figure.Figure, np.ndarray]:
                - density map이 오버레이된 matplotlib Figure 객체
                - RGB 형식의 시각화된 이미지 배열 (H, W, 3)
        Raises:
            ValueError: density_map 또는 processed_image가 None인 경우 (predict 메서드가 실행되지 않은 경우)
        """
        if self.density_map is None or self.processed_image is None:
            raise ValueError("먼저 predict 메서드를 실행하여 예측을 수행해야 합니다.")
        
        fig, ax = plt.subplots(dpi=200, frameon=False)
        ax.imshow(self.processed_image)
        ax.imshow(self.density_map, cmap="jet", alpha=alpha)
        ax.axis("off")
        
        if save:
            if save_path is None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = f"crowd_density_{timestamp}.png"
            
            # 여백 제거하고 저장
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=200)
            print(f"Image saved to: {save_path}")
        
        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        image_from_plot = image_from_plot[:,:,:3]  # RGB로 변환
        
        return fig , image_from_plot
    
    def visualize_dots(self, dot_size: int = 20, sigma: float = 1, percentile: float = 97, 
                    save: bool = False, save_path: Optional[str] = None):
        """
        예측된 군중 위치를 점으로 표시하여 시각화합니다.
        
        Args:
            dot_size (int): 점의 크기. 기본값 20
            sigma (float): Gaussian 필터의 sigma 값. 기본값 1
            percentile (float): 임계값으로 사용할 백분위수 (0-100). 기본값 97
            save (bool): 시각화 결과를 이미지로 저장할지 여부. 기본값 False
            save_path (str, optional): 저장할 경로. None일 경우 현재 디렉토리에 자동 생성된 이름으로 저장.
                기본값 None
                
        Returns:
            Tuple[matplotlib.backends.backend_agg.FigureCanvasBase, np.ndarray]: 
                - matplotlib figure의 canvas 객체
                - RGB 형식의 시각화된 이미지 배열 (H, W, 3)
        Raises:
            ValueError: density_map 또는 processed_image가 None인 경우 (predict 메서드가 실행되지 않은 경우)
        """
        if self.density_map is None or self.processed_image is None:
            raise ValueError("먼저 predict 메서드를 실행하여 예측을 수행해야 합니다.")
            
        adjusted_pred_count = int(round(self.count))
        
        fig, ax = plt.subplots(dpi=200, frameon=False)
        ax.imshow(self.processed_image)
        
        filtered_density = gaussian_filter(self.density_map, sigma=sigma)
        
        threshold = np.percentile(filtered_density, percentile)
        candidate_pixels = np.column_stack(np.where(filtered_density >= threshold))
        
        if len(candidate_pixels) > adjusted_pred_count:
            kmeans = KMeans(n_clusters=adjusted_pred_count, random_state=42, n_init=10)
            kmeans.fit(candidate_pixels)
            head_positions = kmeans.cluster_centers_.astype(int)
        else:
            head_positions = candidate_pixels
            
        y_coords, x_coords = head_positions[:, 0], head_positions[:, 1]
        ax.scatter(x_coords, y_coords, 
                    c='red',
                    s=dot_size,
                    alpha=1.0,
                    edgecolors='white',
                    linewidth=1)
        
        ax.axis("off")
        
        if save:
            if save_path is None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = f"crowd_dots_{timestamp}.png"
            
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=200)
            print(f"Image saved to: {save_path}")
        
        # Figure를 numpy 배열로 변환
        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        image_from_plot = image_from_plot[:,:,:3]  # RGB로 변환
        
        # plt.close(fig)
        # return image_from_plot
        return fig.canvas, image_from_plot
    
    def _process_image(self, image: Union[str, np.ndarray]) -> torch.Tensor:
        """
        이미지를 전처리합니다. 이미지 경로, 넘파이 배열, Streamlit UploadedFile 모두 처리 가능합니다.
        
        Args:
            image: 입력 이미지. 다음 형식 중 하나여야 합니다:
                - str: 이미지 파일 경로
                - np.ndarray: (H, W, 3) 형태의 RGB 이미지
                - UploadedFile: Streamlit의 업로드된 파일
                    
        Returns:
            torch.Tensor: 전처리된 이미지 텐서, shape (1, 3, H, W)
            
        Raises:
            ValueError: 지원하지 않는 이미지 형식이 입력된 경우
            Exception: 이미지 파일을 열 수 없는 경우
        """
        to_tensor = ToTensor()
        normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # 원본 이미지 저장
        self.original_image = image
        
        # 입력 타입에 따른 처리
        if isinstance(image, str):
            # 파일 경로인 경우
            with open(image, "rb") as f:
                pil_image = Image.open(f).convert("RGB")
        elif isinstance(image, np.ndarray):
            # 넘파이 배열인 경우
            if image.dtype == np.uint8:
                pil_image = Image.fromarray(image)
            else:
                # float 타입인 경우 [0, 1] 범위로 가정하고 변환
                pil_image = Image.fromarray((image * 255).astype(np.uint8))
        else:
            # Streamlit UploadedFile 또는 기타 파일 객체인 경우
            try:
                pil_image = Image.open(image).convert("RGB")
            except Exception as e:
                raise ValueError(f"지원하지 않는 이미지 형식입니다: {type(image)}") from e
        
        # 텐서 변환 및 정규화
        tensor_image = to_tensor(pil_image)
        normalized_image = normalize(tensor_image)
        batched_image = normalized_image.unsqueeze(0)  # (1, 3, H, W)
        batched_image = batched_image.to(self.device)

        return batched_image
    def _post_process_image(self, image):
        """이미지 후처리를 수행합니다."""
        image = normalize(image, mean=(0., 0., 0.), 
                        std=(1. / self.std[0], 1. / self.std[1], 1. / self.std[2]))
        image = normalize(image, mean=(-self.mean[0], -self.mean[1], -self.mean[2]), 
                        std=(1., 1., 1.))
        processed_image = to_pil_image(image.squeeze(0))
        return processed_image

    @torch.no_grad()
    def predict(self, image: torch.Tensor) -> Image.Image:
        """
        모델 출력 이미지의 후처리를 수행합니다.
        
        Args:
            image (torch.Tensor): 후처리할 이미지 텐서, shape (1, 3, H, W)
            
        Returns:
            PIL.Image.Image: 후처리된 PIL 이미지
            
        Note:
            이미지 텐서에 대해 정규화를 역변환하고 PIL 이미지 형식으로 변환합니다.
            self.mean과 self.std 값을 사용하여 원본 이미지의 스케일로 복원합니다.
        """
        processed_image = self._process_image(image)
        image_height, image_width = processed_image.shape[-2:]
        processed_image = processed_image.to(self.device)
        
        pred_density = sliding_window_predict(self.model, processed_image, 
                                        self.window_size, self.stride)
        pred_count = pred_density.sum().item()
        resized_pred_density = resize_density_map(pred_density, 
                                                (image_height, image_width)).cpu()
        
        self.processed_image = self._post_process_image(processed_image)
        self.density_map = resized_pred_density.squeeze().numpy()
        self.count = pred_count
        
        return pred_count
    
    def crowd_count(self):
        """
        가장 최근 예측의 군중 수를 반환합니다.
        
        Returns:
            float: 예측된 군중 수
            None: 아직 예측이 수행되지 않은 경우
        """
        return self.count
    
    def get_density_map(self):
        """
        가장 최근 예측의 밀도 맵을 반환합니다.
        
        Returns:
            numpy.ndarray: 밀도 맵
            None: 아직 예측이 수행되지 않은 경우
        """
        return self.density_map
    