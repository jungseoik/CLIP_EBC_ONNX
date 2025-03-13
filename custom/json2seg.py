import json 

def get_segmentation_by_id(target_id, json_file="/home/jungseoik/data/PR/CLIP-EBC/assets/seg.json" ):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # annotations 리스트 가져오기
    annotations = data.get("annotations", [])
    
    # annotations 순회하면서 id가 target_id인 항목 찾기
    for ann in annotations:
        if ann.get("id") == target_id:
            return ann.get("segmentation", None)
    
    # 해당 id가 없으면 None 반환
    return None