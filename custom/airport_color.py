from PIL import Image, ImageDraw
from custom.json2seg import get_segmentation_by_id
import random
INCHEON = "/home/jungseoik/data/PR/CLIP-EBC/assets/incheon.jpg"
COLOR_PAIR = {1: '빨간색', 2: '주황색', 3: '노란색', 4: '초록색', 5: '빨간색', 6: '초록색'}

def generate_random_color_pair():
    colors = ['빨간색', '주황색', '노란색', '초록색']
    return {i: random.choice(colors) for i in range(1, 7)}

def create_mask(segmentation, img_size, color):
    mask = Image.new('RGBA', img_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(mask)

    polygon = segmentation[0] 
    points = [(polygon[i], polygon[i+1]) for i in range(0, len(polygon), 2)]
    
    color_map = {
        '빨간색': (255, 0, 0, 128),    
        '주황색': (255, 165, 0, 128),  
        '노란색': (255, 255, 0, 128),  
        '초록색': (0, 255, 0, 128),
        '파란색': (0, 0, 255, 128),
        '보라색': (128, 0, 128, 128)     
    }
    
    draw.polygon(points, fill=color_map[color])
    return mask

def create_all_masks(img_size, region_color_pairs):
    """
    Parameters:
    - img_size: 이미지 크기
    - region_color_pairs: Dictionary 형태로 {region_id: color} 매핑
        예: {1: '빨간색', 2: '초록색', 3: '노란색', ...}
    """
    # 최종 마스크 생성
    final_mask = Image.new('RGBA', img_size, (0, 0, 0, 0))
    
    # 입력받은 region_color_pairs에 따라 마스크 생성 및 합성
    for region_id, color in region_color_pairs.items():
        segmentation = get_segmentation_by_id(target_id=region_id)
        region_mask = create_mask(segmentation, img_size, color)
        final_mask = Image.alpha_composite(final_mask, region_mask)
    
    return final_mask

def airport_map_color(color_pairs = COLOR_PAIR):
    # region_color_pairs = COLOR_PAIR
    region_color_pairs = generate_random_color_pair()
    image = Image.open(INCHEON)
    all_masks = create_all_masks(image.size, region_color_pairs)
    result = Image.alpha_composite(image.convert('RGBA'), all_masks)
    return result
