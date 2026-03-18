import streamlit as st
from PIL import Image, ImageOps
import os
import glob
from sam_app import InteractiveSAMApp
from streamlit_shortcuts import add_shortcuts

# --- 상수 정의 ---
IMAGE_DIR = "export_results/images"
LABEL_DIR = "export_results/labels_0"
THUMBNAIL_SIZE = (160, 120)  # 4:3 비율
BORDER_WIDTH = 8             # 테두리 두께
LABELED_COLOR = "#a5cbf0"
UNLABELED_COLOR = "#FFFFFF"
COLS_PER_ROW = 10

# --- Helper 함수들 ---

def get_image_files(path):
    """지정된 경로에서 jpg, png 이미지 파일 목록을 가져옵니다."""
    if not os.path.exists(path):
        return []
    image_patterns = [os.path.join(path, f"*.{ext}") for ext in ["jpg", "jpeg", "png"]]
    image_files = []
    for pattern in image_patterns:
        image_files.extend(glob.glob(pattern))
    return sorted(image_files)

def check_label_exists(image_filename, label_dir):
    """이미지에 해당하는 라벨(.txt) 파일이 있는지 확인합니다."""
    base_name = os.path.splitext(os.path.basename(image_filename))[0]
    label_file = os.path.join(label_dir, f"{base_name}.txt")
    return os.path.exists(label_file)

@st.cache_data
def create_bordered_thumbnail_cached(image_path, has_label):
    """이미지 파일 경로를 받아 테두리가 있는 썸네일을 생성합니다."""
    img = Image.open(image_path)
    img.thumbnail(THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
    border_color = LABELED_COLOR if has_label else UNLABELED_COLOR
    bordered_img = ImageOps.expand(img, border=BORDER_WIDTH, fill=border_color)
    return bordered_img

def update_image_statuses():
    """세션 상태를 최신 파일 시스템 정보로 갱신합니다."""
    st.session_state.image_files = get_image_files(IMAGE_DIR)
    st.session_state.label_statuses = {
        os.path.basename(f): check_label_exists(f, LABEL_DIR)
        for f in st.session_state.image_files
    }
    st.toast("이미지 상태를 성공적으로 갱신했습니다!", icon="🔄")

def select_image(path):
    """선택한 이미지를 세션 상태에 저장."""
    st.session_state.selected_image_path = path
    update_image_statuses()
            # st.session_state.selected_image_path = None
            # st.rerun()

# --- Streamlit 앱 메인 로직 ---
st.set_page_config(layout="wide", page_title="이미지 주석 도구")

# --- UI 스타일링 (CSS 주입) ---
st.markdown("""
<style>
    /* 갤러리 내의 버튼에만 스타일 적용 */
    div[data-testid="stVerticalBlock"] div[data-testid="stButton"] > button {
        padding-left: 0.2rem;
        padding-right: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎨 이미지 주석(Annotation) 선택기")
st.markdown("---")

# 세션 상태 초기화
if 'image_files' not in st.session_state:
    st.session_state.image_files = []
    st.session_state.label_statuses = {}
    st.session_state.selected_image_path = None
    update_image_statuses()

# 선택된 이미지에 대한 액션 표시
app_placeholder = st.container()
if not st.session_state.selected_image_path:
    with app_placeholder:
        st.info("아래 목록에서 작업할 이미지를 선택하세요.")

st.markdown("---")
st.header("이미지 목록")

# 업데이트 버튼
if st.button("🔄 상태 업데이트", key="rerun_button"):
    update_image_statuses()
    st.session_state.selected_image_path = None
    st.rerun()

add_shortcuts(rerun_button="r")

# 이미지 갤러리
if not st.session_state.image_files:
    st.warning(f"`{IMAGE_DIR}` 디렉토리에 이미지 파일이 없습니다. 이미지를 추가해주세요.")
else:
    with st.container(height=600):
        num_images = len(st.session_state.image_files)
        num_rows = (num_images + COLS_PER_ROW - 1) // COLS_PER_ROW

        for i in range(num_rows):
            cols = st.columns(COLS_PER_ROW)
            for j in range(COLS_PER_ROW):
                img_index = i * COLS_PER_ROW + j
                if img_index < num_images:
                    image_path = st.session_state.image_files[img_index]
                    image_basename = os.path.basename(image_path)
                    has_label = st.session_state.label_statuses.get(image_basename, False)

                    with cols[j]:
                        thumbnail = create_bordered_thumbnail_cached(image_path, has_label)
                        st.image(thumbnail, use_container_width=True)

                        display_name = image_basename if len(image_basename) <= 50 else image_basename[:47] + "..."
                        st.button(display_name, key=f"btn_{image_path}", use_container_width=True,
                                  on_click=select_image, args=(image_path,))

# 선택된 이미지가 있으면 InteractiveSAMApp 실행
if st.session_state.selected_image_path:
    with app_placeholder:
        try:
            app = InteractiveSAMApp(st.session_state.selected_image_path)
            app.run()
        except FileNotFoundError as e:
            st.error(e)
            st.session_state.selected_image_path = None
        except Exception as e:
            st.error(f"앱 실행 중 오류가 발생했습니다: {e}")
            st.session_state.selected_image_path = None
