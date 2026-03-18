import streamlit as st
import torch
import numpy as np
import cv2
import os
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import time
import random
import copy
import json
from pathlib import Path

# --- Hugging Face와 연동된 SAM2 Video Predictor 임포트 ---
from sam2.sam2_video_predictor import SAM2VideoPredictor

# --- 초기 설정 및 모델 로드 ---
st.set_page_config(layout="wide", page_title="Interactive Video Tracking App")

@st.cache_resource
def load_sam_model():
    """Hugging Face Hub에서 SAM2 Video Predictor를 로드합니다."""
    st.write("Loading SAM2 model from Hugging Face Hub... (최초 실행 시 시간이 걸릴 수 있습니다)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-large")
    predictor.to(device)
    
    st.write("Model loaded successfully.")
    return predictor, device

predictor, device = load_sam_model()

# --- 세션 상태 초기화 ---
def initialize_session_state():
    if "app_state" not in st.session_state: st.session_state.app_state = "upload"
    if "video_frames" not in st.session_state: st.session_state.video_frames = []
    if "video_name" not in st.session_state: st.session_state.video_name = ""
    if "current_frame_idx" not in st.session_state: st.session_state.current_frame_idx = 0
    if "initial_predictor_state" not in st.session_state: st.session_state.initial_predictor_state = None
    if "all_masks" not in st.session_state: st.session_state.all_masks = {}
    if "objects" not in st.session_state: st.session_state.objects = {}
    if "active_object_id" not in st.session_state: st.session_state.active_object_id = None
    if "next_object_id" not in st.session_state: st.session_state.next_object_id = 1
    if "prompts" not in st.session_state: st.session_state.prompts = {}
    if "cached_frames" not in st.session_state: st.session_state.cached_frames = {}
    if "is_playing" not in st.session_state: st.session_state.is_playing = False
    if "needs_recache" not in st.session_state: st.session_state.needs_recache = False
    if "clear_canvas" not in st.session_state: st.session_state.clear_canvas = False
    if "last_canvas_objects" not in st.session_state: st.session_state.last_canvas_objects = None

def reset_to_upload_screen():
    keys_to_reset = list(st.session_state.keys())
    for key in keys_to_reset:
        del st.session_state[key]
    initialize_session_state()

initialize_session_state()

# --- 헬퍼 함수 ---
def get_video_frames_and_init_predictor(uploaded_file):
    if uploaded_file:
        st.session_state.video_name = Path(uploaded_file.name).stem
        temp_file_path = f"./temp_{uploaded_file.name}"
        with open(temp_file_path, "wb") as f: f.write(uploaded_file.getbuffer())
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            st.session_state.initial_predictor_state = predictor.init_state(temp_file_path)
        cap = cv2.VideoCapture(temp_file_path)
        frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for ret, frame in iter(lambda: cap.read(), (False, None))]
        cap.release()
        os.remove(temp_file_path)
        return frames
    return []

def overlay_mask_on_frame(frame, mask, color):
    mask_bool = mask.astype(bool)
    overlay = frame.copy()
    if not overlay.flags.writeable: overlay = overlay.copy()
    overlay[mask_bool] = color
    return cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

def prepare_display_frame(frame_idx):
    base_frame = st.session_state.video_frames[frame_idx].copy()
    for obj_id, masks in st.session_state.all_masks.items():
        if frame_idx in masks:
            color = st.session_state.objects[obj_id]["color"]
            base_frame = overlay_mask_on_frame(base_frame, masks[frame_idx], color)
    for obj_id, prompts_per_frame in st.session_state.prompts.items():
        if frame_idx in prompts_per_frame:
            prompts_on_frame = prompts_per_frame[frame_idx]
            if prompts_on_frame:
                for i, point in enumerate(prompts_on_frame["points"]):
                    label = prompts_on_frame["labels"][i]
                    prompt_color = (0, 255, 0) if label == 1 else (255, 0, 0)
                    marker = cv2.MARKER_CROSS if label == 1 else cv2.MARKER_CROSS
                    cv2.drawMarker(base_frame, (int(point[0]), int(point[1])), prompt_color, marker, 15, 5)
    return base_frame

#---- 내보내기 함수 ----
def export_results():
    output_dir = Path("export_results")
    # 1. images와 json을 위한 하위 디렉토리 경로 지정
    image_dir = output_dir / "images"
    json_dir = output_dir / "json"

    # 2. 하위 디렉토리 생성
    image_dir.mkdir(exist_ok=True, parents=True)
    json_dir.mkdir(exist_ok=True, parents=True)
    
    # 모든 마스크가 있는 프레임 인덱스 집합 구하기
    all_frame_indices = set()
    for obj_id, masks in st.session_state.all_masks.items():
        all_frame_indices.update(masks.keys())

    if not all_frame_indices:
        st.warning("내보낼 추론 결과가 없습니다.")
        return

    with st.spinner(f"결과를 '{output_dir}'에 저장하는 중..."):
        for frame_idx in sorted(list(all_frame_indices)):
            base_filename = f"{st.session_state.video_name}_{frame_idx:05d}"
            
            # 3. 원본 이미지를 'images' 디렉토리에 저장
            original_frame_rgb = st.session_state.video_frames[frame_idx]
            original_frame_bgr = cv2.cvtColor(original_frame_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(image_dir / f"{base_filename}.jpg"), original_frame_bgr)
            
            # 4. JSON 파일을 'json' 디렉토리에 저장
            labels_data = {}
            for obj_id, masks in st.session_state.all_masks.items():
                if frame_idx in masks:
                    mask = masks[frame_idx]
                    mask_uint8 = mask.astype(np.uint8) * 255
                    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    contour_list = [c.squeeze().tolist() for c in contours]
                    obj_name = st.session_state.objects[obj_id]["name"].replace(" ", "_")
                    labels_data[obj_name] = contour_list
            
            with open(json_dir / f"{base_filename}.json", 'w') as f:
                json.dump(labels_data, f, indent=4)

    st.success(f"'{output_dir}'의 images, json 폴더에 {len(all_frame_indices)}개 프레임 결과 저장 완료!")


# --- UI 및 핵심 로직 ---
def draw_sidebar():
    with st.sidebar:
        st.header("🎛️ 객체 관리")
        if st.button("✚ 새 객체 추가", use_container_width=True):
            obj_id = st.session_state.next_object_id
            color = tuple(random.choices(range(100, 256), k=3))
            st.session_state.objects[obj_id] = {"name": f"Object {obj_id}", "color": color}
            st.session_state.active_object_id = obj_id
            st.session_state.next_object_id += 1
            st.rerun()
        if st.session_state.objects:
            obj_names = {obj_id: data["name"] for obj_id, data in st.session_state.objects.items()}
            active_obj_index = list(obj_names.keys()).index(st.session_state.active_object_id) if st.session_state.active_object_id in obj_names else 0
            selected_obj_name = st.radio("편집할 객체 선택:", options=obj_names.values(), index=active_obj_index, key=f"radio_{st.session_state.active_object_id}")
            st.session_state.active_object_id = next(id for id, name in obj_names.items() if name == selected_obj_name)
        
        st.divider()
        st.header("🕹️ 제어")
        play_button_text = "⏹️ 정지" if st.session_state.is_playing else "▶️ 재생"
        if st.button(play_button_text, use_container_width=True, disabled=(not st.session_state.video_frames or len(st.session_state.cached_frames) != len(st.session_state.video_frames))):
            st.session_state.is_playing = not st.session_state.is_playing
            st.rerun()
        if len(st.session_state.cached_frames) != len(st.session_state.video_frames):
             st.caption("Tip: 원활한 재생을 위해 슬라이더를 끝까지 한번 움직여 모든 프레임을 캐싱하세요.")

        # ✨--- 새로운 기능: 내보내기 버튼 ---✨
        if st.button("💾 결과 내보내기", use_container_width=True):
            export_results()

        st.divider()
        if st.button("🔄 처음부터 다시 시작", use_container_width=True):
            reset_to_upload_screen()
            st.rerun()

def draw_main_view():
    if not st.session_state.video_frames: return
    if st.session_state.needs_recache:
        with st.spinner("결과 이미지 캐싱 중..."):
            st.session_state.cached_frames.clear()
            for i in range(len(st.session_state.video_frames)):
                st.session_state.cached_frames[i] = prepare_display_frame(i)
        st.session_state.needs_recache = False
        st.rerun()

    frame_idx = st.slider("타임라인", 0, len(st.session_state.video_frames) - 1, st.session_state.current_frame_idx, key="timeline_slider")
    if frame_idx != st.session_state.current_frame_idx:
        st.session_state.current_frame_idx = frame_idx
        st.session_state.is_playing = False
        st.session_state.clear_canvas = True
    initial_drawing = None
    if st.session_state.clear_canvas:
        initial_drawing = {"version": "4.4.0", "objects": []}
        st.session_state.clear_canvas = False
    display_frame = st.session_state.cached_frames.get(frame_idx, prepare_display_frame(frame_idx))
    st.info("점을 추가/삭제한 후, 아래 버튼으로 마스크를 생성/업데이트하세요.")
    edit_mode = st.radio("프롬프트 모드:", ["Add (추가)", "Remove (제거)"], horizontal=True, key="edit_mode")
    stroke_color = "#00FF00" if "Add" in edit_mode else "#FF0000"
    canvas_result = st_canvas(
        fill_color="rgba(0,0,0,0)", stroke_width=0, stroke_color=stroke_color,
        background_image=Image.fromarray(display_frame), update_streamlit=True,
        height=display_frame.shape[0], width=display_frame.shape[1], 
        drawing_mode="point", point_display_radius=0, key="canvas",
        initial_drawing=initial_drawing
    )
    if canvas_result.json_data is not None and canvas_result.json_data.get("objects"):
        if st.session_state.get("last_canvas_objects") != canvas_result.json_data.get("objects"):
            if st.session_state.active_object_id:
                obj_id = st.session_state.active_object_id
                prompts_for_frame = st.session_state.prompts.setdefault(obj_id, {}).setdefault(frame_idx, {"points": [], "labels": []})
                new_points, new_labels = [], []
                for obj in canvas_result.json_data["objects"]:
                    if obj["type"] == "circle":
                        new_labels.append(1 if obj['stroke'] == '#00FF00' else 0)
                        new_points.append([obj['left'], obj['top']])
                if new_points:
                    prompts_for_frame["points"].extend(new_points)
                    prompts_for_frame["labels"].extend(new_labels)
                    if frame_idx in st.session_state.cached_frames:
                        del st.session_state.cached_frames[frame_idx]
                    st.session_state.last_canvas_objects = canvas_result.json_data.get("objects")
                    st.rerun()

    if st.button("🚀 마스크 생성 / 업데이트", use_container_width=True):
        obj_id = st.session_state.active_object_id
        if not obj_id: st.warning("먼저 객체를 추가하고 선택해주세요."); st.stop()
        prompts_for_obj = st.session_state.prompts.get(obj_id, {})
        valid_prompts = {k: v for k, v in prompts_for_obj.items() if v and v["points"]}
        if not valid_prompts: st.warning("추적을 시작하려면 하나 이상의 프레임에 점을 찍어주세요."); st.stop()
        with st.spinner(f"'{st.session_state.objects[obj_id]['name']}' 추적 중..."):
            inference_state = copy.deepcopy(st.session_state.initial_predictor_state)
            st.session_state.all_masks.setdefault(obj_id, {}).clear()
            sorted_prompt_frames = sorted(valid_prompts.keys())
            for prompt_frame_idx in sorted_prompt_frames:
                prompt = valid_prompts[prompt_frame_idx]
                predictor.add_new_points_or_box(
                    inference_state=inference_state, frame_idx=prompt_frame_idx,
                    obj_id=obj_id, points=prompt["points"], labels=prompt["labels"]
                )
            for f_idx, obj_ids, masks in predictor.propagate_in_video(inference_state):
                if obj_id in obj_ids:
                    obj_output_idx = obj_ids.index(obj_id)
                    mask = masks[0, obj_output_idx].cpu().numpy() > 0.0
                    st.session_state.all_masks[obj_id][f_idx] = mask
            st.session_state.needs_recache = True
            st.success("작업이 완료되었습니다!")
            st.rerun()

# --- 메인 앱 로직 ---
if st.session_state.app_state == "upload":
    st.title("🎬 Interactive Video Object Tracking App")
    uploaded_file = st.file_uploader("추적할 비디오 파일을 업로드하세요.", type=["mp4", "avi", "mov"])
    if uploaded_file:
        with st.spinner("비디오 처리 및 모델 초기화 중..."):
            st.session_state.video_frames = get_video_frames_and_init_predictor(uploaded_file)
        st.session_state.app_state = "edit"
        st.rerun()
elif st.session_state.app_state == "edit":
    draw_sidebar()
    st.title("✍️ Edit and Track")
    draw_main_view()
    if st.session_state.is_playing:
        if st.session_state.current_frame_idx < len(st.session_state.video_frames) - 1:
            st.session_state.current_frame_idx += 1
        else:
            st.session_state.is_playing = False
        time.sleep(1/30)
        st.rerun()