import cv2
import numpy as np
import os
import multiprocessing
import streamlit as st
import json # JSON 파일을 다루기 위해 추가
from pathlib import Path # 파일 경로를 다루기 위해 추가

# -----------------------------------------------------------------------------
# 1. JSON 라벨을 처리하는 내부 클래스 (SAM 로직 대체됨)
# -----------------------------------------------------------------------------
class _JSONLabelProcessor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.original_image = self.load_image(image_path)
        self.display_image = self.original_image.copy()
        self.image_basename = os.path.basename(image_path)

        # JSON 라벨 데이터 로드 및 처리
        self.objects_data = self.load_json_labels(image_path)
        if not self.objects_data:
            # JSON 파일이 없거나 비어있으면 경고 메시지를 이미지에 표시
            cv2.putText(self.original_image, f"No label file found for {self.image_basename}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            self.display_image = self.original_image.copy()

        # 상태 변수 초기화
        self.app_mode = "segment"  # 'a' 키를 누르면 이 모드가 됨
        self.last_mask = None
        self.polygon = None
        
        # 'q', 'e', 'r' 키 기능 관련 변수
        self.q_point = None
        self.e_point = None
        self.r_click_point = None
        self.surrounding_points = []
        self.surrounding_points_by_num = {}
        self.yolo_annotations = {i: [] for i in range(6)}
        self.polygon_annotations = []

        self.WINDOW_NAME = f"Interactive Labeling: {self.image_basename}"
        cv2.namedWindow(self.WINDOW_NAME)
        cv2.setMouseCallback(self.WINDOW_NAME, self.mouse_callback)

    def load_image(self, path):
        image = cv2.imread(path)
        if image is None:
            print(f"{path} 파일을 찾을 수 없습니다. 임시 이미지로 대체합니다.")
            image = np.zeros((600, 800, 3), dtype=np.uint8)
            cv2.putText(image, f"{path} not found", (100, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return image

    def load_json_labels(self, image_path):
        """이미지 경로 기준으로 ../json/ 에서 라벨 파일을 로드합니다."""
        p = Path(image_path)
        # image_path의 부모 디렉토리(images)의 부모 디렉토리(../)로 이동한 후 'json' 폴더 경로를 만듭니다.
        json_dir = p.parent.parent / "json"
        # 이미지 파일명(확장자 제외)에 .json을 붙여 최종 경로를 완성합니다.
        json_path = json_dir / f"{p.stem}.json"
        
        objects = {}
        if not json_path.exists():
            print(f"경고: 라벨 파일을 찾을 수 없습니다: {json_path}")
            return objects

        with open(json_path, 'r') as f:
            data = json.load(f)
        
        for obj_name, contours_list in data.items():
            if not contours_list: continue
            main_contour_list = max(contours_list, key=lambda c: cv2.contourArea(np.array(c, dtype=np.int32)))
            contour = np.array(main_contour_list, dtype=np.int32)
            bbox = cv2.boundingRect(contour)
            objects[obj_name] = {'contour': contour, 'bbox': bbox}
            print(f"객체 '{obj_name}' 로드 완료. Bbox: {bbox}")
            
        return objects

    def mouse_callback(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        click_point = (x, y)

        if self.app_mode == "segment":
            selected_obj_contour = None
            for obj_name, data in self.objects_data.items():
                bx, by, bw, bh = data['bbox']
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    selected_obj_contour = data['contour']
                    print(f"객체 '{obj_name}' 선택됨.")
                    break
            
            if selected_obj_contour is not None:
                # 1. 선택된 객체의 컨투어로 마스크 생성 (기존 'a' 동작)
                mask = np.zeros(self.original_image.shape[:2], dtype=np.uint8)
                cv2.drawContours(mask, [selected_obj_contour], -1, 255, thickness=cv2.FILLED)
                self.last_mask = mask
                
                # ✨--- 수정: 'f' 키 동작을 여기에 통합 ---✨
                # 2. 생성된 마스크로 즉시 폴리곤 저장
                self.save_polygon()
                
                self.app_mode = "edit_polygon"
                print("객체 선택 및 폴리곤 저장이 완료되었습니다. 'q'와 'e' 키로 점을 지정하세요.")
                # self.save_polygon() 내부에서 self.update_display()가 호출되므로 별도 호출 불필요
            else:
                print("객체가 없는 영역을 클릭했습니다.")

        # ... 이 아래의 elif self.app_mode in ["set_q", "set_e"] 등은 변경 없음 ...
        elif self.app_mode in ["set_q", "set_e"]:
            if self.polygon is None:
                print("'a' 키를 누른 후 객체를 먼저 선택하세요."); return
            closest = self.find_closest_point_on_polygon(self.polygon, click_point)
            if self.app_mode == "set_q":
                self.q_point = closest; self.e_point = None; self.surrounding_points = []
                print(f"시작(q)점 설정: {self.q_point}")
            elif self.app_mode == "set_e":
                self.e_point = closest; self.surrounding_points = []
                print(f"중간(e)점 설정: {self.e_point}")
            self.app_mode = "edit_polygon"
            self.update_display()

        elif self.app_mode == "set_r":
            self.r_click_point = click_point
            print(f"'r' 클릭 위치: {self.r_click_point}")
            
            self.surrounding_points_by_num = {}
            for num in range(6):
                surrounding_pts = self.calculate_surrounding_points(self.polygon, self.q_point, self.e_point, num=num)
                final_points_tuple = tuple(surrounding_pts) + (self.r_click_point,)
                self.surrounding_points_by_num[num] = final_points_tuple
            
            self.surrounding_points = self.calculate_surrounding_points(self.polygon, self.q_point, self.e_point, num=3)
            
            print(f"주변점 계산 완료. {len(self.surrounding_points_by_num)}개의 num 세트 저장됨.")
            self.app_mode = "edit_polygon"
            self.update_display()

        elif self.app_mode == "set_r":
            self.r_click_point = click_point
            print(f"'r' 클릭 위치: {self.r_click_point}")
            
            self.surrounding_points_by_num = {}
            for num in range(6):
                surrounding_pts = self.calculate_surrounding_points(self.polygon, self.q_point, self.e_point, num=num)
                final_points_tuple = tuple(surrounding_pts) + (self.r_click_point,)
                self.surrounding_points_by_num[num] = final_points_tuple
            
            self.surrounding_points = self.calculate_surrounding_points(self.polygon, self.q_point, self.e_point, num=3)
            
            print(f"주변점 계산 완료. {len(self.surrounding_points_by_num)}개의 num 세트 저장됨.")
            self.app_mode = "edit_polygon"
            self.update_display()

    def update_display(self):
        """화면에 표시될 이미지를 현재 상태에 맞게 갱신합니다."""
        self.display_image = self.original_image.copy()
        if self.last_mask is not None:
            overlay = np.zeros_like(self.original_image)
            overlay[self.last_mask > 0] = (0, 255, 0) # 초록색 마스크
            self.display_image = cv2.addWeighted(self.original_image, 0.9, overlay, 0.1, 0)
        
        self.draw_visual_elements()

    def draw_visual_elements(self):
        """폴리곤, q, e, r 점 등 시각적 요소를 그립니다."""
        if self.polygon is not None:
            cv2.polylines(self.display_image, [self.polygon], True, (255, 255, 0), 2)
        if self.q_point: cv2.circle(self.display_image, self.q_point, 2, (255, 0, 0), -1)
        if self.e_point: cv2.circle(self.display_image, self.e_point, 2, (0, 255, 255), -1)
        if self.r_click_point: cv2.circle(self.display_image, self.r_click_point, 2, (0, 165, 255), -1)
        for pt in self.surrounding_points:
            if pt not in [self.q_point, self.e_point]:
                cv2.circle(self.display_image, pt, 3, (255, 0, 255), -1)

    # --- 이 아래의 복잡한 계산 함수들은 원본과 동일하게 유지 ---
    def find_closest_point_on_polygon(self, poly, clicked_point):
        poly_points = poly.reshape(-1, 2)
        distances = np.linalg.norm(poly_points - clicked_point, axis=1)
        return tuple(poly_points[np.argmin(distances)])

    def extract_side_points(self, poly_points, indices, offset_distance, num):
        if len(indices) < 2: return []
        seg_lens, new_pts = [0], []
        for i in range(1, len(indices)):
            p1, p2 = poly_points[indices[i-1]], poly_points[indices[i]]
            seg_lens.append(seg_lens[-1] + np.linalg.norm(p2 - p1))
        total_len = seg_lens[-1]
        if total_len == 0: return []
        targets = [total_len * i / (num + 1) for i in range(1, num + 1)]
        for pos in targets:
            for i in range(1, len(seg_lens)):
                if seg_lens[i-1] <= pos <= seg_lens[i]:
                    seg_len = seg_lens[i] - seg_lens[i-1]
                    r = (pos - seg_lens[i-1]) / seg_len if seg_len > 0 else 0
                    p1, p2 = poly_points[indices[i-1]], poly_points[indices[i]]
                    interp = p1 + r * (p2 - p1)
                    tangent = p2 - p1
                    normal = np.array([tangent[1], -tangent[0]], dtype=np.float32)
                    norm = np.linalg.norm(normal) + 1e-8
                    normal /= norm
                    new_pts.append(tuple((interp + normal * offset_distance).astype(int)))
                    break
        return new_pts

    def calculate_surrounding_points(self, poly, start_pt, end_pt, num, offset_distance=0):
        poly_points = poly.reshape(-1, 2)
        n = len(poly_points)
        s_idx = np.argmin(np.linalg.norm(poly_points - start_pt, axis=1))
        e_idx = np.argmin(np.linalg.norm(poly_points - end_pt, axis=1))
        path_cw = self.get_path_indices(n, s_idx, e_idx, True)
        path_ccw = self.get_path_indices(n, s_idx, e_idx, False)
        len_cw = self.calculate_path_length(poly_points, path_cw)
        len_ccw = self.calculate_path_length(poly_points, path_ccw)
        long_path, short_path = (path_cw, path_ccw) if len_cw >= len_ccw else (path_ccw, path_cw)
        long_pts = self.extract_side_points(poly_points, long_path, offset_distance, num=num)
        short_pts = self.extract_side_points(poly_points, short_path, offset_distance, num=num)
        if num == 0:
            return [start_pt] + long_pts + [end_pt] + short_pts[::-1]
        vec_start2end = (end_pt[0] - start_pt[0], end_pt[1] - start_pt[1])
        v_start2p = (long_pts[0][0] - start_pt[0], long_pts[0][1] - start_pt[1])
        cross_product = vec_start2end[0] * v_start2p[1] - vec_start2end[1] * v_start2p[0]
        if cross_product > 0:
            return [start_pt] + long_pts + [end_pt] + short_pts[::-1]
        else:
            return [start_pt] + short_pts + [end_pt] + long_pts[::-1]

    def calculate_path_length(self, poly_points, indices):
        if len(indices) < 2: return 0
        return sum(np.linalg.norm(poly_points[indices[i]] - poly_points[indices[i+1]]) for i in range(len(indices)-1))

    def get_path_indices(self, n, s_idx, e_idx, clockwise=True):
        path, curr = [], s_idx
        while curr != e_idx:
            path.append(curr)
            curr = (curr + 1 if clockwise else curr - 1 + n) % n
        path.append(e_idx)
        return path

    def save_polygon(self):
        if self.last_mask is not None:
            contours, _ = cv2.findContours(self.last_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                original_polygon = max(contours, key=cv2.contourArea)
                refined_points = []
                points = original_polygon.reshape(-1, 2)
                num_points = len(points)
                for i in range(num_points):
                    p1 = points[i]
                    p2 = points[(i + 1) % num_points]
                    refined_points.append(p1)
                    distance = np.linalg.norm(p1 - p2)
                    if distance > 4:
                        for j in range(1, int(distance)//2):
                            new_point = p1 + (p2 - p1) * j /(int(distance)//2)
                            refined_points.append(new_point.astype(int))
                    elif distance > 2:
                        new_point = p1 + (p2 - p1) * 0.5
                        refined_points.append(new_point.astype(int))
                self.polygon = np.array(refined_points, dtype=np.int32).reshape(-1, 1, 2)
                self.q_point = self.e_point = self.r_click_point = None
                self.surrounding_points = []
                print(f"폴리곤 세분화 및 저장 완료! {len(self.polygon)}개의 점.")
                self.update_display()
    
    def generate_yolo_annotations(self):
        if not self.surrounding_points_by_num or self.polygon is None:
            print("'f'로 폴리곤 저장 후 'r' 모드로 주변점을 먼저 계산하세요.")
            return
        img_h, img_w, _ = self.original_image.shape
        x, y, w, h = cv2.boundingRect(self.polygon)
        offset = 5
        x, y, w, h = x - offset, y - offset, w + offset*2, h + offset*2
        x_center_norm = (x + w / 2) / img_w
        y_center_norm = (y + h / 2) / img_h
        w_norm = w / img_w
        h_norm = h / img_h
        bbox_str = f"{x_center_norm:.6f} {y_center_norm:.6f} {w_norm:.6f} {h_norm:.6f}"
        poly_norm = self.polygon.reshape(-1, 2).astype(np.float32)
        poly_norm[:, 0] /= img_w
        poly_norm[:, 1] /= img_h
        poly_str = " ".join([f"{p[0]:.6f} {p[1]:.6f}" for p in poly_norm])
        self.polygon_annotations.append(poly_str)
        for num, points_tuple in self.surrounding_points_by_num.items():
            keypoints_parts = []
            for pt in points_tuple:
                x_norm, y_norm, visibility = pt[0] / img_w, pt[1] / img_h, 2
                keypoints_parts.append(f"{x_norm:.6f} {y_norm:.6f} {visibility}")
            keypoints_str = " ".join(keypoints_parts)
            full_yolo_str = f"0 {bbox_str} {keypoints_str}"
            self.yolo_annotations[num].append(full_yolo_str)
        self.display_image = self.original_image.copy()
        print(f"'t' 눌림: YOLO 어노테이션 {len(self.surrounding_points_by_num)}개 생성 및 저장 준비 완료.")

    def save_all_annotations(self):
        base_name = os.path.splitext(self.image_basename)[0]
        for num, annotations in self.yolo_annotations.items():
            if annotations:
                save_dir = f"export_results/labels_{num}"; os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"{base_name}.txt")
                with open(save_path, "w") as f: f.write("\n".join(annotations))
                print(f"'{save_path}'에 키포인트 어노테이션 저장 완료.")
        if self.polygon_annotations:
            save_dir = "export_results/contours"; os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{base_name}.txt")
            with open(save_path, "w") as f: f.write("\n".join(self.polygon_annotations))
            print(f"'{save_path}'에 컨투어 어노테이션 저장 완료.")

    def reset_state(self):
        self.last_mask = self.polygon = self.q_point = self.e_point = self.r_click_point = None
        self.surrounding_points = []; self.surrounding_points_by_num = {}
        self.yolo_annotations = {i: [] for i in range(6)}; self.polygon_annotations = []
        self.app_mode = "segment"
        self.update_display()
        print("모든 포인트, 폴리곤, 어노테이션 초기화")

    def run(self):
        print("--- 사용법 ---"); print(" a:객체선택모드 | f:폴리곤저장 | x:초기화 | q:시작점 e:끝점 r:주변점계산 | t:어노테이션생성 | ESC:저장&종료"); print("---")
        self.update_display()
        while True:
            cv2.imshow(self.WINDOW_NAME, self.display_image)
            key = cv2.waitKey(1) & 0xFF
            if key == 27: self.save_all_annotations(); break
            elif key == ord('a'): self.app_mode = "segment"; print("모드: 객체 선택 (마우스 클릭)")
            elif key == ord('x'): self.reset_state()
            elif key == ord('z'): print("'z' (되돌리기) 기능은 제거되었습니다.")
            elif key == ord('q'): self.app_mode = "set_q"; print("모드: 시작점(q) 설정 대기")
            elif key == ord('e'): self.app_mode = "set_e"; print("모드: 중간점(e) 설정 대기")
            elif key == ord('r'):
                if self.q_point and self.e_point: self.app_mode = "set_r"; print("모드: 주변점 계산(r) 대기")
                else: print("먼저 q와 e 점을 지정하세요.")
            elif key == ord('t'): self.generate_yolo_annotations()
        cv2.destroyAllWindows()

# -----------------------------------------------------------------------------
# 2. 멀티프로세스를 실행하는 함수
# -----------------------------------------------------------------------------
def _run_app_in_new_process(image_path):
    print(f"새 프로세스 시작: {image_path}")
    try:
        app = _JSONLabelProcessor(image_path=image_path)
        app.run()
    except Exception as e:
        print(f"프로세스에서 오류 발생: {e}")
    print(f"새 프로세스 종료: {image_path}")

# -----------------------------------------------------------------------------
# 3. Streamlit이 직접 호출할 공개 클래스 (변경 없음)
# -----------------------------------------------------------------------------
class InteractiveSAMApp:
    def __init__(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")
        self.image_path = image_path
    def run(self):
        st.info(f"`{os.path.basename(self.image_path)}`에 대한 상호작용 라벨링 작업을 새 창에서 시작합니다.")
        process = multiprocessing.Process(target=_run_app_in_new_process, args=(self.image_path,))
        process.start()
        st.success(f"라벨링 창이 실행 중입니다. 새 창을 확인하세요. 해당 창에서 'ESC' 키를 누르면 종료됩니다.")