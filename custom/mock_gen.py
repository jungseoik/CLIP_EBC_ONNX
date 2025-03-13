import pandas as pd
import numpy as np
import random
def create_mock_data_heatmap():
    # 기본 구조 생성
    sections = [f'구역 {i}' for i in range(1, 7)]
    months = list(range(1, 13))
    years = list(range(2020, 2025))
    
    # 데이터프레임용 리스트 생성
    data = []
    for year in years:
        for section in sections:
            for month in months:
                data.append({
                    'section': section,
                    'month': month,
                    'year': year,
                    'crowd_count': np.random.randint(30000, 500000)
                })
    
    # DataFrame 생성
    df = pd.DataFrame(data)
    return df

def create_mock_data_table():
    mock_data = {
    'section': [f'구역 {i}' for i in range(1, 7)],
    'count': np.random.randint(10000, 300000, 6)  
}

    df = pd.DataFrame(mock_data)
    return df

def create_mock_data_donut(min_value=10000, max_value=500000):
    """
    가상의 인구 이동 데이터를 생성합니다.
    
    Returns:
        tuple: (인바운드 이동 비율, 아웃바운드 이동 비율)
    """
    # 랜덤 값 생성 (10000~500000 사이)
    inbound = random.randint(min_value, max_value)
    outbound = random.randint(min_value, max_value)
    
    # 전체 값 대비 비율 계산 (0-100 사이의 값으로 변환)
    total = inbound + outbound
    inbound_percent = round((inbound / total) * 100)
    outbound_percent = round((outbound / total) * 100)
    
    return inbound_percent, outbound_percent


def create_mock_data_inout():
    """
    방문객 데이터 랜덤 생성
    - 이번달 방문객: 150,000 ~ 500,000
    - 오늘 방문객: 5,000 ~ 100,000
    - delta는 전월/전일 대비 증감량 (-30% ~ +30%)
    """
    # 이번달 방문객 (더 큰 범위)
    monthly_visitors = random.randint(150000, 500000)
    monthly_delta = int(monthly_visitors * random.uniform(-0.3, 0.3))  # 30% 범위 내 증감
    
    # 오늘 방문객 (더 작은 범위)
    daily_visitors = random.randint(5000, 100000)
    daily_delta = int(daily_visitors * random.uniform(-0.3, 0.3))  # 30% 범위 내 증감
    
    return {
        'top': {
            'state': '이번달 방문객',
            'visitor': monthly_visitors,
            'delta': monthly_delta
        },
        'bottom': {
            'state': '오늘 방문객',
            'visitor': daily_visitors,
            'delta': daily_delta
        }
    }