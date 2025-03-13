import altair as alt
import pandas as pd
from typing import Tuple, Literal, Union
# Heatmap
def make_heatmap(input_df, input_y, input_x, input_color, input_color_theme):
    heatmap = alt.Chart(input_df).mark_rect().encode(
            y=alt.Y(f'{input_y}:O', axis=alt.Axis(title="Month", titleFontSize=18, titlePadding=15, titleFontWeight=900, labelAngle=0)),
            x=alt.X(f'{input_x}:O', axis=alt.Axis(title="", titleFontSize=18, titlePadding=15, titleFontWeight=900, labelAngle=0)),
            color=alt.Color(f'max({input_color}):Q',
                             legend=None,
                             scale=alt.Scale(scheme=input_color_theme)),
            stroke=alt.value('black'),
            strokeWidth=alt.value(0.25),
        ).properties(width=900
        ).configure_axis(
        labelFontSize=12,
        titleFontSize=12
        ) 
    # height=300
    return heatmap


# Donut chart
def make_donut(
    input_response: float,
    input_text: str,
    input_color: Literal['blue', 'green', 'orange', 'red']
) -> alt.LayerChart:
    """
    Altair를 사용하여 지정된 퍼센트, 레이블, 색상 스키마로 도넛 차트를 생성합니다.

    함수 구조:
    1. 입력 색상에 따른 색상 스키마 정의
    2. 두 개의 DataFrame 생성:
        - 퍼센트 표시를 위한 메인 데이터
        - 전체 원을 위한 배경 데이터
    3. 세 개의 레이어 생성:
        - 배경 원 (plot_bg)
        - 퍼센트 호 (plot)
        - 중앙 텍스트 표시

    매개변수:
    ----------
    input_response : float
        표시할 퍼센트 값 (0-100 사이)
    input_text : str
        차트에 표시할 레이블 텍스트
    input_color : str
        사용할 색상 스키마 ('blue', 'green', 'orange', 'red' 중 하나)

    반환값:
    -------
    alt.LayerChart
        배경, 퍼센트 호, 중앙 텍스트가 결합된 Altair 레이어 차트

    사용 예시:
    ---------
    >>> chart = make_donut(75, "완료", "blue")
    >>> chart.save('donut.html')
    """
    if input_color == 'blue':
        chart_color = ['#29b5e8', '#155F7A']
    if input_color == 'green':
        chart_color = ['#27AE60', '#12783D']
    if input_color == 'orange':
        chart_color = ['#F39C12', '#875A12']
    if input_color == 'red':
        chart_color = ['#E74C3C', '#781F16']

    source = pd.DataFrame({
        "Topic": ['', input_text],
        "% value": [100-input_response, input_response]
    })
    source_bg = pd.DataFrame({
        "Topic": ['', input_text],
        "% value": [100, 0]
    })

    plot = alt.Chart(source).mark_arc(innerRadius=45, cornerRadius=25).encode(
        theta="% value",
        color= alt.Color("Topic:N",
                        scale=alt.Scale(
                            #domain=['A', 'B'],
                            domain=[input_text, ''],
                            # range=['#29b5e8', '#155F7A']),  # 31333F
                            range=chart_color),
                        legend=None),
    ).properties(width=130, height=130)

    text = plot.mark_text(align='center', color="#29b5e8", font="Lato", fontSize=32, fontWeight=700, fontStyle="italic").encode(text=alt.value(f'{input_response} %'))
    plot_bg = alt.Chart(source_bg).mark_arc(innerRadius=45, cornerRadius=20).encode(
        theta="% value",
        color= alt.Color("Topic:N",
                        scale=alt.Scale(
                            # domain=['A', 'B'],
                            domain=[input_text, ''],
                            range=chart_color),  # 31333F
                        legend=None),
    ).properties(width=130, height=130)
    return plot_bg + plot + text