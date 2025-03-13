
import streamlit as st
import altair as alt
from custom.airport_color import airport_map_color
from custom.mock_gen import create_mock_data_heatmap, create_mock_data_table, create_mock_data_donut, create_mock_data_inout
from custom.visual import make_heatmap, make_donut
from custom.clip_ebc import ClipEBC
import os
import torch
import assets
from custom.clip_ebc import ClipEBC

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#######################
# Page configuration
st.set_page_config(
    page_title="Incheon Airport Dashboard Mockup",
    page_icon="🛫",
    layout="wide",
    initial_sidebar_state="expanded")

alt.theme.enable("dark")
#######################
# Sidebar
with st.sidebar:
    st.title('🛫Incheon Airport Dashboard Mockup')

    selected_year = 2017

    color_theme_list = ['blues', 'cividis', 'greens', 'inferno', 'magma', 'plasma', 'reds', 'rainbow', 'turbo', 'viridis']
    selected_color_theme = st.selectbox('Select a color theme', color_theme_list)

    st.subheader("Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        model = ClipEBC()
        count = model.predict(uploaded_file)
        _ , clip_image = model.visualize_dots(save=False)
        st.image(clip_image, caption='Uploaded Image', use_container_width=True)
        st.metric(label="군중수", value=count)
#######################
# Convert population to text 
def format_number(num):
    if num > 1000000:
        if not num % 1000000:
            return f'{num // 1000000} M'
        return f'{round(num / 1000000, 1)} M'
    return f'{num // 1000} K'

#######################
# Dashboard Main Panel
col = st.columns((1.5, 4.5, 2), gap='medium')

with col[0]:
    st.markdown('#### Gains/Losses')
    mock_data = create_mock_data_inout()
    # Display top state metric
    st.metric(
        label=mock_data['top']['state'],
        value=format_number(mock_data['top']['visitor']),
        delta=format_number(mock_data['top']['delta'])
    )
    # Display bottom state metric
    st.metric(
        label=mock_data['bottom']['state'],
        value=format_number(mock_data['bottom']['visitor']),
        delta=format_number(mock_data['bottom']['delta'])
    )
####################### 도넛 파트
    header = st.empty()
    selected_area = st.selectbox(
        '구역 선택',
        options=range(1, 7),
        format_func=lambda x: f'{x}구역'
    )
    header.markdown(f'#### {selected_area}구역 혼잡상황')
    inbound_migration, outbound_migration = create_mock_data_donut()
    donut_chart_greater = make_donut(inbound_migration, 'Inbound Migration', 'green')
    donut_chart_less = make_donut(outbound_migration, 'Outbound Migration', 'red')

    migrations_col = st.columns((0.2, 1, 0.2))
    with migrations_col[1]:
        st.write('Inbound')
        st.altair_chart(donut_chart_greater)
        st.write('Outbound')
        st.altair_chart(donut_chart_less)

with col[1]:
    st.markdown('#### 전체 혼잡 상황')
    result = airport_map_color()
    st.image(result, caption='AIR PORT',  use_container_width=True)
    df = create_mock_data_heatmap()
    selected_year = st.selectbox(
        "표시할 연도 선택",
        options=sorted(df['year'].unique(), reverse=True),  # 최신 연도가 먼저 나오도록
        index=0  # 가장 최신 연도를 기본값으로
    )
    df_selected = df[df['year'] == selected_year].copy()
    heatmap = make_heatmap(df_selected, 'month', 'section', 'crowd_count', selected_color_theme)
    st.altair_chart(heatmap,use_container_width=True)

with col[2]:
    st.markdown('#### RealTime Counting ')
    df = create_mock_data_table()
    df_sorted = df.sort_values(by="section", ascending=True)
    st.dataframe(df,
                column_order=("section", "count"),
                hide_index=True,
                width=None,
                column_config={
                    "section": st.column_config.TextColumn(
                        "구역",
                    ),
                    "count": st.column_config.ProgressColumn(
                        "현재 추정",
                        format="%d",  
                        min_value=0,
                        max_value=max(df["count"]),
                    )}
                )
    
    st.markdown(f'#### {selected_area} 구역 CCTV')
    st.video("/home/jungseoik/data/PR/CLIP-EBC/assets/sample_airport.mp4", autoplay=True)
