import os
import sys
import torch
import numpy as np
import tensorrt as trt
from typing import Union, Tuple, Optional
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor, Normalize
from torchvision.transforms.functional import normalize, to_pil_image
import json
import datetime
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans
import assets

# 프로젝트 루트 디렉토리 설정
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

class ClipEBCTensorRT:
    """
    CLIP-EBC (Efficient Boundary Counting) TensorRT 버전 이미지 처리 클래스입니다.
    
    TensorRT로 변환된 CLIP 모델을 사용하여 이미지를 처리하며, 슬라이딩 윈도우 예측 기능을 포함한
    다양한 설정 옵션을 제공합니다.
    """
    
    def __init__(self,
                 engine_path="assets/CLIP_EBC_nwpu_rmse_tensorrt.trt",
                 truncation=4,
                 reduction=8,
                 granularity="fine",
                 anchor_points="average",
                 input_size=224,
                 window_size=224,
                 stride=224,
                 dataset_name="qnrf",
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225),
                 config_dir ="configs"):
        """CLIPEBC TensorRT 클래스를 설정 매개변수와 함께 초기화합니다."""
        self.engine_path = engine_path
        self.truncation = truncation
        self.reduction = reduction
        self.granularity = granularity
        self.anchor_points_type = anchor_points
        self.input_size = input_size
        self.window_size = window_size
        self.stride = stride
        self.dataset_name = dataset_name
        self.mean = mean
        self.std = std
        self.config_dir = config_dir
        
        # 결과 저장용 변수 초기화
        self.density_map = None
        self.processed_image = None
        self.count = None
        self.original_image = None
        
        # TensorRT 엔진 로드
        print(f"TensorRT 엔진 로드 중: {self.engine_path}")
        self._load_engine()
        
        # 입력 및 출력 이름 설정
        self.input_name = "input"
        self.output_name = "output"
        
        print(f"TensorRT 엔진 초기화 완료")
        
    def _load_engine(self):
        """TensorRT 엔진을 로드합니다."""
        # TensorRT 로거 생성
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        
        # 런타임 생성
        self.runtime = trt.Runtime(TRT_LOGGER)
        
        # 엔진 파일 로드
        with open(self.engine_path, 'rb') as f:
            engine_data = f.read()
        
        # 직렬화된 엔진에서 엔진 생성
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        
        # 실행 컨텍스트 생성
        self.context = self.engine.create_execution_context()
        
        # TensorRT 10.x에서는 input_binding/output_binding 대신 네트워크 구조를 확인
        # 입력과 출력을 가져오는 방법이 변경됨
        self.num_io_tensors = self.engine.num_io_tensors
        
        # 입력과 출력 텐서 이름 찾기
        self.input_tensor_names = []
        self.output_tensor_names = []
        
        print(f"TensorRT 엔진에서 {self.num_io_tensors}개의 IO 텐서를 찾았습니다")
        
        for i in range(self.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            
            if is_input:
                self.input_tensor_names.append(name)
            else:
                self.output_tensor_names.append(name)
        
        # 입력과 출력 이름 설정
        if not self.input_tensor_names:
            raise ValueError("엔진에서 입력 텐서를 찾을 수 없습니다.")
        if not self.output_tensor_names:
            raise ValueError("엔진에서 출력 텐서를 찾을 수 없습니다.")
        
        # 기본 입력 및 출력 이름 설정
        self.input_name = self.input_tensor_names[0]
        self.output_name = self.output_tensor_names[0]
        
        # 입출력 형태 추출
        self.input_shape = self.engine.get_tensor_shape(self.input_name)
        self.output_shape = self.engine.get_tensor_shape(self.output_name)
        
        print(f"입력 이름: {self.input_name}, 형태: {self.input_shape}")
        print(f"출력 이름: {self.output_name}, 형태: {self.output_shape}")
    
    def _process_image(self, image: Union[str, np.ndarray]) -> np.ndarray:
        """
        이미지를 전처리합니다. 이미지 경로, 넘파이 배열, Streamlit UploadedFile 모두 처리 가능합니다.
        
        Args:
            image: 입력 이미지. 다음 형식 중 하나여야 합니다:
                - str: 이미지 파일 경로
                - np.ndarray: (H, W, 3) 형태의 RGB 이미지
                - UploadedFile: Streamlit의 업로드된 파일
                    
        Returns:
            np.ndarray: 전처리된 이미지 배열, shape (1, 3, H, W)
        """
        to_tensor = ToTensor()
        normalize = Normalize(mean=self.mean, std=self.std)
        
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
        
        # numpy로 변환
        numpy_image = batched_image.numpy()
        
        return numpy_image
    
    def _post_process_image(self, image_tensor):
        """이미지 텐서를 PIL 이미지로 변환합니다."""
        # NumPy 배열을 PyTorch 텐서로 변환
        if isinstance(image_tensor, np.ndarray):
            image_tensor = torch.from_numpy(image_tensor)
            
        # 정규화 역변환
        image = normalize(
            image_tensor,
            mean=[0., 0., 0.],
            std=[1./self.std[0], 1./self.std[1], 1./self.std[2]]
        )
        
        image = normalize(
            image,
            mean=[-self.mean[0], -self.mean[1], -self.mean[2]],
            std=[1., 1., 1.]
        )
        
        # 배치 차원 제거 및 PIL 이미지로 변환
        processed_image = to_pil_image(image.squeeze(0))
        return processed_image
    def _infer_batch(self, batch_input):
        """
        TensorRT 엔진을 사용하여 배치 추론을 수행합니다. (수정 버전)
        """
        import pycuda.driver as cuda
        import pycuda.autoinit
        import numpy as np
        
        batch_size = batch_input.shape[0]
        
        # 입력의 형태와 데이터 타입 확인
        input_shape = (batch_size, 3, self.input_size, self.input_size)
        print(f"입력 배치 형태: {batch_input.shape}, 데이터 타입: {batch_input.dtype}")
        
        # 입력 형태 검증
        if batch_input.shape != input_shape:
            print(f"경고: 입력 형태 불일치. 예상: {input_shape}, 실제: {batch_input.shape}")
            # 필요시 형태 수정
            batch_input = np.resize(batch_input, input_shape)
        
        # 데이터 타입 검증
        if batch_input.dtype != np.float32:
            print(f"경고: 입력 데이터 타입 불일치. float32로 변환합니다.")
            batch_input = batch_input.astype(np.float32)
        
        # 동적 배치 크기 설정
        self.context.set_input_shape(self.input_name, input_shape)
        
        # 출력 형태 가져오기
        output_shape = self.context.get_tensor_shape(self.output_name)
        output_shape = tuple(output_shape)  # 튜플로 변환하여 안전성 보장
        print(f"출력 형태: {output_shape}")
        
        # -1 값을 실제 배치 크기로 대체
        if output_shape[0] == -1:
            output_shape = (batch_size,) + output_shape[1:]
        
        # 출력 버퍼 준비
        output = np.empty(output_shape, dtype=np.float32)
        
        # 호스트 메모리 준비 (페이지 잠금 메모리 사용)
        h_input = cuda.pagelocked_empty(batch_input.shape, dtype=np.float32)
        h_output = cuda.pagelocked_empty(output_shape, dtype=np.float32)
        
        # 입력 데이터 복사
        np.copyto(h_input, batch_input)
        
        # 디바이스 메모리 할당
        d_input = cuda.mem_alloc(h_input.nbytes)
        d_output = cuda.mem_alloc(h_output.nbytes)
        
        # CUDA 스트림 생성
        stream = cuda.Stream()
        
        try:
            # 메모리 복사 (호스트 -> 디바이스)
            cuda.memcpy_htod_async(d_input, h_input, stream)
            
            # 텐서 주소 설정
            self.context.set_tensor_address(self.input_name, int(d_input))
            self.context.set_tensor_address(self.output_name, int(d_output))
            
            # 디버깅 정보 (메모리 주소)
            print(f"입력 메모리 주소: {int(d_input)}, 출력 메모리 주소: {int(d_output)}")
            
            # 실행
            success = self.context.execute_async_v3(stream_handle=stream.handle)
            if not success:
                print("TensorRT 실행 실패")
                return None
            
            # 메모리 복사 (디바이스 -> 호스트)
            cuda.memcpy_dtoh_async(h_output, d_output, stream)
            
            # 스트림 동기화
            stream.synchronize()
            
            # 출력 데이터 복사
            np.copyto(output, h_output)
            
            return output
            
        except Exception as e:
            print(f"TensorRT 추론 중 오류 발생: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
            
        finally:
            # 메모리 해제
            del stream
            if 'd_input' in locals():
                d_input.free()
            if 'd_output' in locals():
                d_output.free()

    def sliding_window_predict(self, image: np.ndarray, window_size: Union[int, Tuple[int, int]], 
                             stride: Union[int, Tuple[int, int]]) -> np.ndarray:
        """
        슬라이딩 윈도우 방식으로 이미지 예측을 수행합니다. 겹치는 영역은 평균값을 사용합니다.
        
        Args:
            image (np.ndarray): 형태가 (1, 3, H, W)인 이미지 배열
            window_size (int or tuple): 윈도우 크기
            stride (int or tuple): 윈도우 이동 간격
            
        Returns:
            np.ndarray: 예측된 밀도 맵
        """
        # CUDA 초기화 (처음 사용할 때만)
        global cuda
        if 'cuda' not in globals():
            import pycuda.driver as cuda
            cuda.init()
        
        # 입력 검증
        assert len(image.shape) == 4, f"이미지는 4차원 배열이어야 합니다. (1, C, H, W), 현재: {image.shape}"
        
        # 윈도우 크기와 스트라이드 설정
        window_size = (int(window_size), int(window_size)) if isinstance(window_size, (int, float)) else window_size
        stride = (int(stride), int(stride)) if isinstance(stride, (int, float)) else stride
        window_size = tuple(window_size)
        stride = tuple(stride)
        
        # 검증
        assert isinstance(window_size, tuple) and len(window_size) == 2 and window_size[0] > 0 and window_size[1] > 0, \
            f"윈도우 크기는 양수 정수 튜플 (h, w)이어야 합니다. 현재: {window_size}"
        assert isinstance(stride, tuple) and len(stride) == 2 and stride[0] > 0 and stride[1] > 0, \
            f"스트라이드는 양수 정수 튜플 (h, w)이어야 합니다. 현재: {stride}"
        assert stride[0] <= window_size[0] and stride[1] <= window_size[1], \
            f"스트라이드는 윈도우 크기보다 작아야 합니다. 현재: {stride}와 {window_size}"
        
        image_height, image_width = image.shape[-2:]
        window_height, window_width = window_size
        stride_height, stride_width = stride
        
        # 슬라이딩 윈도우 수 계산
        num_rows = int(np.ceil((image_height - window_height) / stride_height) + 1)
        num_cols = int(np.ceil((image_width - window_width) / stride_width) + 1)
        
        # 윈도우 추출
        windows = []
        window_positions = []
        for i in range(num_rows):
            for j in range(num_cols):
                x_start, y_start = i * stride_height, j * stride_width
                x_end, y_end = x_start + window_height, y_start + window_width
                
                # 이미지 경계 처리
                if x_end > image_height:
                    x_start, x_end = image_height - window_height, image_height
                if y_end > image_width:
                    y_start, y_end = image_width - window_width, image_width
                
                window = image[:, :, x_start:x_end, y_start:y_end]
                windows.append(window)
                window_positions.append((x_start, y_start, x_end, y_end))
        
        # 배치 단위로 추론
        all_preds = []
        max_batch_size = 8
        
        for start_idx in range(0, len(windows), max_batch_size):
            end_idx = min(start_idx + max_batch_size, len(windows))
            batch_windows = np.vstack(windows[start_idx:end_idx])  # (batch_size, 3, h, w)
            
            # TensorRT 추론
            batch_preds = self._infer_batch(batch_windows)
            
            # Debug 정보
            # print(f"배치 입력 형태: {batch_windows.shape}, 배치 출력 형태: {batch_preds.shape}")
            
            all_preds.extend([batch_preds[i:i+1] for i in range(batch_preds.shape[0])])
        
        # 예측 결과를 numpy 배열로 변환
        preds = np.concatenate(all_preds, axis=0)
        
        # 출력 밀도 맵 조립
        pred_map = np.zeros((preds.shape[1], image_height // self.reduction, image_width // self.reduction), dtype=np.float32)
        count_map = np.zeros((preds.shape[1], image_height // self.reduction, image_width // self.reduction), dtype=np.float32)
        
        idx = 0
        for i in range(num_rows):
            for j in range(num_cols):
                x_start, y_start, x_end, y_end = window_positions[idx]
                
                # 출력 영역 계산 (reduction 고려)
                x_start_out = x_start // self.reduction
                y_start_out = y_start // self.reduction
                x_end_out = x_end // self.reduction
                y_end_out = y_end // self.reduction
                
                pred_map[:, x_start_out:x_end_out, y_start_out:y_end_out] += preds[idx]
                count_map[:, x_start_out:x_end_out, y_start_out:y_end_out] += 1.
                idx += 1
        
        # 겹치는 영역 평균 계산
        pred_map /= count_map
        
        return pred_map

    def resize_density_map(self, density_map: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        밀도 맵의 크기를 조정합니다. 총합은 보존됩니다.
        
        Args:
            density_map: 형태가 (C, H, W)인 밀도 맵
            target_size: 목표 크기 (H', W')
            
        Returns:
            np.ndarray: 크기가 조정된 밀도 맵
        """
        from PIL import Image
        import torch.nn.functional as F
        import torch
        
        # numpy를 torch로 변환
        if isinstance(density_map, np.ndarray):
            density_map = torch.from_numpy(density_map)
        
        # 배치 차원 추가
        if density_map.dim() == 3:
            density_map = density_map.unsqueeze(0)  # (1, C, H, W)
        
        current_size = density_map.shape[2:]
        
        if current_size[0] == target_size[0] and current_size[1] == target_size[1]:
            return density_map.squeeze(0).numpy()
        
        # 원본 밀도 맵의 총합 계산
        original_sum = density_map.sum()
        
        # 크기 조정 (쌍선형 보간)
        resized_map = F.interpolate(
            density_map,
            size=target_size,
            mode='bilinear',
            align_corners=False
        )
        
        # 총합 보존을 위한 스케일링
        if resized_map.sum() > 0:  # 0으로 나누기 방지
            resized_map = resized_map * (original_sum / resized_map.sum())
        
        return resized_map.squeeze(0).numpy()

    def predict(self, image: Union[str, np.ndarray]) -> float:
        """
        이미지에서 군중 계수 예측을 수행합니다.
        
        Args:
            image: 입력 이미지 (경로, 넘파이 배열, 또는 업로드된 파일)
            
        Returns:
            float: 예측된 사람 수
        """
        # 이미지 전처리
        processed_image = self._process_image(image)
        image_height, image_width = processed_image.shape[-2:]
        
        # 슬라이딩 윈도우 예측
        pred_density = self.sliding_window_predict(
            processed_image, 
            self.window_size, 
            self.stride
        )
        
        # 예측 결과 저장
        pred_count = pred_density.sum()
        
        # 원본 이미지 크기로 밀도 맵 조정
        resized_pred_density = self.resize_density_map(
            pred_density, 
            (image_height, image_width)
        )
        
        # 결과 저장
        self.processed_image = self._post_process_image(processed_image)
        self.density_map = resized_pred_density.squeeze()
        self.count = pred_count
        
        return pred_count
    
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
        """
        if self.density_map is None or self.processed_image is None:
            raise ValueError("먼저 predict 메서드를 실행하여 예측을 수행해야 합니다.")
        
        fig, ax = plt.subplots(dpi=200, frameon=False)
        ax.imshow(self.processed_image)
        ax.imshow(self.density_map, cmap="jet", alpha=alpha)
        ax.axis("off")
        plt.title(f"Count: {self.count:.1f}")
        
        if save:
            if save_path is None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = f"crowd_density_{timestamp}.png"
            
            # 여백 제거하고 저장
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=200)
            print(f"이미지 저장 완료: {save_path}")
        
        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        image_from_plot = image_from_plot[:,:,:3]  # RGB로 변환
        
        return fig, image_from_plot
    
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
        plt.title(f"Count: {self.count:.1f}")
        
        if save:
            if save_path is None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = f"crowd_dots_{timestamp}.png"
            
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=200)
            print(f"이미지 저장 완료: {save_path}")
        
        # Figure를 numpy 배열로 변환
        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        image_from_plot = image_from_plot[:,:,:3]  # RGB로 변환
        
        return fig.canvas, image_from_plot
    
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