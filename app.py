import gradio as gr
from custom.clip_ebc_onnx import ClipEBCOnnx
from custom.clip_ebc import ClipEBC

import numpy as np
import matplotlib.pyplot as plt

# ONNX 모델 초기화
model = ClipEBC()

def predict_crowd(image):
    """
    이미지를 받아서 군중 수를 예측하고 시각화 결과를 반환합니다.
    
    Args:
        image: Gradio에서 받은 이미지 (numpy array)
        
    Returns:
        tuple: (예측된 군중 수, 밀도 맵 시각화, 점 시각화)
    """
    count = model.predict(image)
    
    # 밀도 맵 시각화
    fig_density, density_map = model.visualize_density_map()
    plt.close(fig_density)  # 메모리 누수 방지
    # 점 시각화
    canvas, dot_map = model.visualize_dots()
    if canvas is not None:
        plt.close(canvas.figure)
    else:
        dot_map = np.zeros_like(density_map)

    return (
        f"예측된 군중 수: {count:.1f}명",
        density_map,
        dot_map
    )

with gr.Blocks(title="CLIP-EBC Crowd Counter") as app:
    gr.Markdown("# CLIP-EBC Crowd Counter")
    gr.Markdown("이미지를 업로드하여 군중 수를 예측하고 시각화합니다.")
    
    with gr.Row():
        input_image = gr.Image(type="numpy", label="입력 이미지")
    
    with gr.Row():
        predict_btn = gr.Button("예측", variant="primary")
    
    with gr.Row():
        count_text = gr.Textbox(label="예측 결과")
    
    with gr.Row():
        with gr.Column():
            density_output = gr.Image(label="밀도 맵")
        with gr.Column():
            dots_output = gr.Image(label="점 시각화")
    
    predict_btn.click(
        fn=predict_crowd,
        inputs=input_image,
        outputs=[count_text, density_output, dots_output]
    )

# user_name = "piaspace"
# user_pw = "piaspace@418"
# auth_pair = [(user_name, user_pw)]
if __name__ == "__main__":
    # app.launch(server_name = "0.0.0.0", server_port= 7860,share=False , auth=auth_pair)
    app.launch(server_name = "0.0.0.0", server_port= 7860,share=False )
