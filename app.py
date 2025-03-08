import cv2
import numpy as np
import mediapipe as mp
import time
import random
import os
import yaml
import math
from collections import deque

# ===============================================================
# Classe do Pac-Man
# ===============================================================
class PacManGame:
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        
        # Configurações de desempenho
        self.reduce_resolution = True  # Reduzir resolução para melhorar desempenho
        self.process_every_n_frame = 2  # Processar apenas 1 a cada N frames
        self.frame_count = 0
        
        # Tamanho da grade do jogo
        self.grid_size = 20
        self.cols = width // self.grid_size
        self.rows = height // self.grid_size
        
        # Inicializar o labirinto (0 = caminho, 1 = parede)
        self.maze = np.zeros((self.rows, self.cols), dtype=np.uint8)
        self.generate_simple_maze()
        
        # Posição do Pac-Man
        self.pacman_pos = [self.cols // 2, self.rows // 2]
        self.pacman_dir = [0, 0]  # [x, y]
        
        # Comida
        self.foods = []
        self.generate_foods(30)  # Gerar 30 comidas iniciais
        
        # Fantasmas
        self.ghosts = []
        self.generate_ghosts(3)  # Gerar 3 fantasmas
        
        # Pontuação
        self.score = 0
        
        # Estado do jogo
        self.game_over = False
        self.win = False
        
        # Carregar sons (verificar se existe e usar se possível)
        self.has_sound = self._try_import_sound()
        self.eat_sound = None
        self.death_sound = None
        
        # Histórico de posições da mão para tornar o movimento mais suave
        self.hand_positions = []
        self.max_hand_history = 5

        # Inicializa MediaPipe Hands para desenho (opcional, para desenhar na tela)
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def _try_import_sound(self):
        try:
            import pygame
            pygame.mixer.init()
            if os.path.exists('pacman_beginning.wav'):
                pygame.mixer.music.load('pacman_beginning.wav')
            
            if os.path.exists('pacman_chomp.wav'):
                self.eat_sound = pygame.mixer.Sound('pacman_chomp.wav')
            
            if os.path.exists('pacman_death.wav'):
                self.death_sound = pygame.mixer.Sound('pacman_death.wav')
            
            return True
        except (ImportError, ModuleNotFoundError):
            print("Som não disponível. Pygame não instalado ou arquivos de som ausentes.")
            return False
        except Exception as e:
            print(f"Erro ao configurar som: {e}")
            return False

    def generate_simple_maze(self):
        # Criar uma grade simples com alguns obstáculos
        # Bordas
        self.maze[0, :] = 1
        self.maze[-1, :] = 1
        self.maze[:, 0] = 1
        self.maze[:, -1] = 1
        
        # Blocos aleatórios
        for _ in range(20):
            x = random.randint(2, self.cols - 3)
            y = random.randint(2, self.rows - 3)
            size = random.randint(2, 4)
            self.maze[y:y+size, x:x+size] = 1
            
        # Garantir que o centro esteja vazio para o Pac-Man
        center_x, center_y = self.cols // 2, self.rows // 2
        self.maze[center_y-1:center_y+2, center_x-1:center_x+2] = 0

    def generate_foods(self, count):
        for _ in range(count):
            while True:
                x = random.randint(1, self.cols - 2)
                y = random.randint(1, self.rows - 2)
                if self.maze[y, x] == 0 and [x, y] not in self.foods and [x, y] != self.pacman_pos:
                    self.foods.append([x, y])
                    break

    def generate_ghosts(self, count):
        for _ in range(count):
            while True:
                x = random.randint(1, self.cols - 2)
                y = random.randint(1, self.rows - 2)
                # Posicionar fantasmas longe do Pac-Man inicial
                if (self.maze[y, x] == 0 and 
                    abs(x - self.pacman_pos[0]) > 5 and 
                    abs(y - self.pacman_pos[1]) > 5):
                    self.ghosts.append({
                        'pos': [x, y],
                        'dir': [random.choice([-1, 0, 1]), random.choice([-1, 0, 1])],
                        'color': (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    })
                    break

    def update_pacman_direction(self, hand_x, hand_y):
        # Normalizar para o centro da tela
        center_x, center_y = self.width // 2, self.height // 2
        
        # Calcular a direção relativa da mão para o centro
        dx = hand_x - center_x
        dy = hand_y - center_y
        
        # Aplicar um limiar para evitar movimentos acidentais
        threshold = 50
        if abs(dx) < threshold and abs(dy) < threshold:
            self.pacman_dir = [0, 0]
            return
        
        # Adicionar posição ao histórico
        self.hand_positions.append((dx, dy))
        if len(self.hand_positions) > self.max_hand_history:
            self.hand_positions.pop(0)
        
        # Calcular a média das últimas posições para suavizar o movimento
        avg_dx = sum(pos[0] for pos in self.hand_positions) / len(self.hand_positions)
        avg_dy = sum(pos[1] for pos in self.hand_positions) / len(self.hand_positions)
        
        # Decidir a direção principal (horizontal ou vertical)
        if abs(avg_dx) > abs(avg_dy):
            self.pacman_dir = [1 if avg_dx > 0 else -1, 0]
        else:
            self.pacman_dir = [0, 1 if avg_dy > 0 else -1]

    def update(self):
        if self.game_over or self.win:
            return
            
        # Mover Pac-Man
        new_x = self.pacman_pos[0] + self.pacman_dir[0]
        new_y = self.pacman_pos[1] + self.pacman_dir[1]
        
        # Verificar colisão com paredes
        if 0 <= new_x < self.cols and 0 <= new_y < self.rows and self.maze[new_y, new_x] == 0:
            self.pacman_pos = [new_x, new_y]
        
        # Verificar colisão com comida
        for food in self.foods[:]:
            if self.pacman_pos == food:
                self.foods.remove(food)
                self.score += 10
                if self.has_sound and self.eat_sound is not None:
                    try:
                        self.eat_sound.play()
                    except Exception as e:
                        print(f"Erro ao reproduzir som: {e}")
                        self.has_sound = False
                if len(self.foods) < 10:
                    self.generate_foods(5)
        
        # Mover fantasmas
        for ghost in self.ghosts:
            if random.random() < 0.05:
                ghost['dir'] = [random.choice([-1, 0, 1]), random.choice([-1, 0, 1])]
                if ghost['dir'] == [0, 0]:
                    ghost['dir'] = [random.choice([-1, 1]), 0]
            
            new_x = ghost['pos'][0] + ghost['dir'][0]
            new_y = ghost['pos'][1] + ghost['dir'][1]
            
            if not (0 <= new_x < self.cols and 0 <= new_y < self.rows and self.maze[new_y, new_x] == 0):
                ghost['dir'] = [random.choice([-1, 0, 1]), random.choice([-1, 0, 1])]
                if ghost['dir'] == [0, 0]:
                    ghost['dir'] = [random.choice([-1, 1]), 0]
                continue
            
            ghost['pos'] = [new_x, new_y]
            
            if ghost['pos'] == self.pacman_pos:
                self.game_over = True
                if self.has_sound and self.death_sound is not None:
                    try:
                        self.death_sound.play()
                    except Exception as e:
                        print(f"Erro ao reproduzir som: {e}")
                        self.has_sound = False

    def draw(self, frame):
        # Criar um canvas para o jogo
        game_canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Desenhar o labirinto
        for y in range(self.rows):
            for x in range(self.cols):
                if self.maze[y, x] == 1:
                    pt1 = (x * self.grid_size, y * self.grid_size)
                    pt2 = ((x + 1) * self.grid_size, (y + 1) * self.grid_size)
                    cv2.rectangle(game_canvas, pt1, pt2, (0, 0, 255), -1)
        
        # Desenhar comida
        for food in self.foods:
            center = (food[0] * self.grid_size + self.grid_size // 2, 
                      food[1] * self.grid_size + self.grid_size // 2)
            cv2.circle(game_canvas, center, self.grid_size // 4, (255, 255, 0), -1)
        
        # Desenhar fantasmas
        for ghost in self.ghosts:
            center = (ghost['pos'][0] * self.grid_size + self.grid_size // 2, 
                      ghost['pos'][1] * self.grid_size + self.grid_size // 2)
            cv2.circle(game_canvas, center, self.grid_size // 2, ghost['color'], -1)
            eye_offset = self.grid_size // 6
            left_eye = (center[0] - eye_offset, center[1] - eye_offset)
            right_eye = (center[0] + eye_offset, center[1] - eye_offset)
            cv2.circle(game_canvas, left_eye, self.grid_size // 8, (255, 255, 255), -1)
            cv2.circle(game_canvas, right_eye, self.grid_size // 8, (255, 255, 255), -1)
            cv2.circle(game_canvas, left_eye, self.grid_size // 16, (0, 0, 0), -1)
            cv2.circle(game_canvas, right_eye, self.grid_size // 16, (0, 0, 0), -1)
        
        # Desenhar Pac-Man
        center = (self.pacman_pos[0] * self.grid_size + self.grid_size // 2, 
                  self.pacman_pos[1] * self.grid_size + self.grid_size // 2)
        mouth_angle = int(30 * abs(np.sin(time.time() * 10)))
        start_angle = 0
        if self.pacman_dir[0] > 0:
            start_angle = 0
        elif self.pacman_dir[0] < 0:
            start_angle = 180
        elif self.pacman_dir[1] > 0:
            start_angle = 90
        elif self.pacman_dir[1] < 0:
            start_angle = 270
            
        cv2.ellipse(game_canvas, center, (self.grid_size // 2, self.grid_size // 2), 
                   0, start_angle + mouth_angle, start_angle + 360 - mouth_angle, (0, 255, 255), -1)
        
        # Desenhar pontuação
        cv2.putText(game_canvas, f"Score: {self.score}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Mensagem de Game Over
        if self.game_over:
            cv2.putText(game_canvas, "GAME OVER", (self.width // 2 - 100, self.height // 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
            cv2.putText(game_canvas, "Pressione 'R' para reiniciar", (self.width // 2 - 150, self.height // 2 + 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Combinar o jogo com o frame da câmera
        alpha = 0.7
        cv2.addWeighted(game_canvas, alpha, frame, 1 - alpha, 0, frame)
        
        return frame

    def reset(self):
        self.pacman_pos = [self.cols // 2, self.rows // 2]
        self.pacman_dir = [0, 0]
        self.foods = []
        self.generate_foods(30)
        self.ghosts = []
        self.generate_ghosts(3)
        self.score = 0
        self.game_over = False
        self.win = False
        self.hand_positions = []

# ===============================================================
# Classe AR Virtual Assistant (com integração do Pac-Man)
# ===============================================================
class ARVirtualAssistant:
    def __init__(self, config_path='config.yaml'):
        # Carrega configurações
        if os.path.exists(config_path):
            with open(config_path, 'r') as file:
                self.config = yaml.safe_load(file)
        else:
            # Configuração padrão caso o arquivo não exista
            self.config = {
                'video_source': 0,
                'min_detection_confidence': 0.7,
                'min_tracking_confidence': 0.5,
                'face_min_detection_confidence': 0.5,
                'face_min_tracking_confidence': 0.5
            }

        # Configuração da câmera
        self.cap = cv2.VideoCapture(self.config.get('video_source', 0))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Configurações MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Inicializa detectores
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=self.config.get('min_detection_confidence', 0.7),
            min_tracking_confidence=self.config.get('min_tracking_confidence', 0.5)
        )

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=self.config.get('face_min_detection_confidence', 0.5),
            min_tracking_confidence=self.config.get('face_min_tracking_confidence', 0.5)
        )

        # Configurações de desenho
        self.hand_drawing_spec = self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
        self.face_drawing_spec = self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1)
        
        # Estado do aplicativo
        self.mode = "menu"  # modos: menu, drawing, filters, game
        self.drawing_color = (0, 0, 255)
        self.drawing_thickness = 5
        self.drawings = []
        self.current_drawing = deque(maxlen=1000)
        self.is_drawing = False
        
        # Elementos de interface virtual
        self.menu_buttons = [
            {"name": "Desenhar", "rect": (50, 50, 150, 100), "color": (0, 150, 0), "action": "drawing"},
            {"name": "Filtros", "rect": (220, 50, 150, 100), "color": (150, 0, 0), "action": "filters"},
            {"name": "Jogo", "rect": (390, 50, 150, 100), "color": (0, 0, 150), "action": "game"}
        ]
        
        # Inicialmente, nenhuma instância do jogo Pac-Man
        self.pacman_game = None
        
        # Filtros faciais
        self.current_filter = None
        self.filters = ["none", "neon", "cartoon", "mask"]
        self.filter_index = 0
        
        # Rastreamento de gestos
        self.pinch_active = False
        self.last_pinch_pos = None
        self.gesture_history = deque(maxlen=10)
        
        # Carrega assets
        self.load_assets()

    def load_assets(self):
        try:
            self.hat_img = cv2.imread('assets/hat.png', cv2.IMREAD_UNCHANGED)
            self.glasses_img = cv2.imread('assets/glasses.png', cv2.IMREAD_UNCHANGED)
            self.mask_img = cv2.imread('assets/mask.png', cv2.IMREAD_UNCHANGED)
        except Exception as e:
            print(f"Erro ao carregar assets: {e}")
            self.hat_img = np.zeros((100, 100, 4), dtype=np.uint8)
            self.glasses_img = np.zeros((50, 150, 4), dtype=np.uint8)
            self.mask_img = np.zeros((100, 100, 4), dtype=np.uint8)

    def overlay_image(self, background, foreground, x, y, scale=1.0):
        if scale != 1.0:
            foreground = cv2.resize(foreground, (0, 0), fx=scale, fy=scale)
            
        h, w = foreground.shape[:2]
        if x < 0: x = 0
        if y < 0: y = 0
        
        if y + h > background.shape[0] or x + w > background.shape[1]:
            h_visible = min(h, background.shape[0] - y)
            w_visible = min(w, background.shape[1] - x)
            if h_visible <= 0 or w_visible <= 0:
                return background
            foreground = foreground[:h_visible, :w_visible]
            h, w = foreground.shape[:2]
        
        roi = background[y:y+h, x:x+w]
        
        if foreground.shape[2] == 4:
            alpha = foreground[:, :, 3] / 255.0
            alpha = np.expand_dims(alpha, axis=2)
            rgb = foreground[:, :, :3]
            blended = (1 - alpha) * roi + alpha * rgb
            background[y:y+h, x:x+w] = blended
        else:
            background[y:y+h, x:x+w] = foreground
            
        return background

    def process_frame(self, image):
        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape
        
        display_image = image.copy()
        
        # Processa mãos
        hand_results = self.hands.process(image_rgb)
        hand_landmarks = []
        if hand_results.multi_hand_landmarks:
            for hand_landmarks_obj in hand_results.multi_hand_landmarks:
                landmarks = []
                for lm in hand_landmarks_obj.landmark:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmarks.append((cx, cy))
                hand_landmarks.append(landmarks)
                if self.mode != "game":
                    self.mp_drawing.draw_landmarks(
                        display_image, 
                        hand_landmarks_obj, 
                        self.mp_hands.HAND_CONNECTIONS,
                        self.hand_drawing_spec, 
                        self.hand_drawing_spec
                    )
        
        # Processa face
        face_results = self.face_mesh.process(image_rgb)
        face_landmarks = None
        if face_results.multi_face_landmarks:
            face_landmarks_obj = face_results.multi_face_landmarks[0]
            face_landmarks = []
            for lm in face_landmarks_obj.landmark:
                cx, cy = int(lm.x * w), int(lm.y * h)
                face_landmarks.append((cx, cy))
            if self.mode == "filters" and self.current_filter == "none":
                self.mp_drawing.draw_landmarks(
                    display_image,
                    face_landmarks_obj,
                    self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
                )
        
        if self.mode == "menu":
            display_image = self.process_menu(display_image, hand_landmarks)
        elif self.mode == "drawing":
            display_image = self.process_drawing(display_image, hand_landmarks)
        elif self.mode == "filters":
            display_image = self.process_filters(display_image, face_landmarks)
        elif self.mode == "game":
            display_image = self.process_game(display_image, hand_landmarks, face_landmarks)
        
        if self.mode != "menu":
            cv2.rectangle(display_image, (20, 20), (120, 70), (50, 50, 50), -1)
            cv2.putText(display_image, "Menu", (35, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            if hand_landmarks and len(hand_landmarks) > 0:
                index_tip = hand_landmarks[0][8]
                if 20 <= index_tip[0] <= 120 and 20 <= index_tip[1] <= 70:
                    self.mode = "menu"
        
        return display_image

    def process_menu(self, image, hand_landmarks):
        cv2.putText(image, "AR Virtual Assistant", (self.width//2 - 150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        for button in self.menu_buttons:
            x, y, w, h = button["rect"]
            cv2.rectangle(image, (x, y), (x+w, y+h), button["color"], -1)
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 255), 2)
            text_size = cv2.getTextSize(button["name"], cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            text_x = x + (w - text_size[0]) // 2
            text_y = y + (h + text_size[1]) // 2
            cv2.putText(image, button["name"], (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        if hand_landmarks and len(hand_landmarks) > 0:
            index_tip = hand_landmarks[0][8]
            for button in self.menu_buttons:
                x, y, w, h = button["rect"]
                if x <= index_tip[0] <= x+w and y <= index_tip[1] <= y+h:
                    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 0), 3)
                    thumb_tip = hand_landmarks[0][4]
                    distance = math.sqrt((thumb_tip[0] - index_tip[0])**2 + (thumb_tip[1] - index_tip[1])**2)
                    if distance < 30:
                        self.mode = button["action"]
                        if button["action"] == "game":
                            self.init_game()
                        elif button["action"] == "filters":
                            self.current_filter = self.filters[self.filter_index]
        return image

    def process_drawing(self, image, hand_landmarks):
        colors = [
            (0, 0, 255), (0, 255, 0), (255, 0, 0),
            (0, 255, 255), (255, 0, 255), (255, 255, 0),
            (255, 255, 255)
        ]
        for i, color in enumerate(colors):
            x = 150 + i * 60
            cv2.rectangle(image, (x, 20), (x + 50, 70), color, -1)
            cv2.rectangle(image, (x, 20), (x + 50, 70), (0, 0, 0), 2)
            if color == self.drawing_color:
                cv2.rectangle(image, (x-2, 18), (x+52, 72), (0, 0, 0), 3)
        cv2.rectangle(image, (self.width - 120, 20), (self.width - 20, 70), (100, 100, 100), -1)
        cv2.putText(image, "Limpar", (self.width - 110, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        for drawing in self.drawings:
            points = list(drawing["points"])
            for i in range(1, len(points)):
                cv2.line(image, points[i-1], points[i], drawing["color"], drawing["thickness"])
        points = list(self.current_drawing)
        for i in range(1, len(points)):
            cv2.line(image, points[i-1], points[i], self.drawing_color, self.drawing_thickness)
        if hand_landmarks and len(hand_landmarks) > 0:
            index_tip = hand_landmarks[0][8]
            for i, color in enumerate(colors):
                x = 150 + i * 60
                if x <= index_tip[0] <= x+50 and 20 <= index_tip[1] <= 70:
                    thumb_tip = hand_landmarks[0][4]
                    distance = math.sqrt((thumb_tip[0] - index_tip[0])**2 + (thumb_tip[1] - index_tip[1])**2)
                    if distance < 30:
                        self.drawing_color = color
            if self.width-120 <= index_tip[0] <= self.width-20 and 20 <= index_tip[1] <= 70:
                thumb_tip = hand_landmarks[0][4]
                distance = math.sqrt((thumb_tip[0] - index_tip[0])**2 + (thumb_tip[1] - index_tip[1])**2)
                if distance < 30:
                    self.drawings = []
                    self.current_drawing = deque(maxlen=1000)
            middle_tip = hand_landmarks[0][12]
            ring_base = hand_landmarks[0][13]
            if middle_tip[1] > ring_base[1]:
                if not self.is_drawing:
                    self.is_drawing = True
                    self.current_drawing = deque(maxlen=1000)
                if index_tip[1] > 80:
                    self.current_drawing.append(index_tip)
            else:
                if self.is_drawing:
                    self.is_drawing = False
                    if len(self.current_drawing) > 1:
                        self.drawings.append({
                            "points": self.current_drawing.copy(),
                            "color": self.drawing_color,
                            "thickness": self.drawing_thickness
                        })
                    self.current_drawing = deque(maxlen=1000)
        return image

    def process_filters(self, image, face_landmarks):
        cv2.rectangle(image, (150, 20), (250, 70), (100, 100, 100), -1)
        cv2.putText(image, "< Prev", (160, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.rectangle(image, (350, 20), (450, 70), (100, 100, 100), -1)
        cv2.putText(image, "Next >", (360, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        filter_name = self.filters[self.filter_index].capitalize()
        cv2.rectangle(image, (260, 20), (340, 70), (50, 50, 150), -1)
        cv2.putText(image, filter_name, (270, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if face_landmarks:
            if self.current_filter == "neon":
                image = self.apply_neon_filter(image, face_landmarks)
            elif self.current_filter == "cartoon":
                image = self.apply_cartoon_filter(image, face_landmarks)
            elif self.current_filter == "mask":
                image = self.apply_mask_filter(image, face_landmarks)
        return image

    def apply_neon_filter(self, image, face_landmarks):
        jaw_points = [i for i in range(0, 17)]
        eyebrow_points = [i for i in range(17, 27)]
        nose_points = [i for i in range(27, 36)]
        mask = np.zeros_like(image)
        for i in range(1, len(jaw_points)):
            p1 = face_landmarks[jaw_points[i-1]]
            p2 = face_landmarks[jaw_points[i]]
            cv2.line(mask, p1, p2, (0, 255, 255), 8)
        for i in range(1, len(eyebrow_points)):
            p1 = face_landmarks[eyebrow_points[i-1]]
            p2 = face_landmarks[eyebrow_points[i]]
            cv2.line(mask, p1, p2, (255, 0, 255), 8)
        for i in range(1, len(nose_points)):
            p1 = face_landmarks[nose_points[i-1]]
            p2 = face_landmarks[nose_points[i]]
            cv2.line(mask, p1, p2, (0, 255, 0), 8)
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        return cv2.addWeighted(image, 0.7, mask, 0.3, 0)

    def apply_cartoon_filter(self, image, face_landmarks):
        left_eye = face_landmarks[159]
        right_eye = face_landmarks[386]
        eye_distance = math.sqrt((left_eye[0] - right_eye[0])**2 + (left_eye[1] - right_eye[1])**2)
        if self.glasses_img is not None:
            glasses_width = int(eye_distance * 2.5)
            glasses_height = int(glasses_width * self.glasses_img.shape[0] / self.glasses_img.shape[1])
            glasses_resized = cv2.resize(self.glasses_img, (glasses_width, glasses_height))
            glasses_x = right_eye[0] - int(glasses_width * 0.25)
            glasses_y = min(right_eye[1], left_eye[1]) - int(glasses_height * 0.5)
            image = self.overlay_image(image, glasses_resized, glasses_x, glasses_y)
        top_head = face_landmarks[10]
        if self.hat_img is not None:
            hat_width = int(eye_distance * 3)
            hat_height = int(hat_width * self.hat_img.shape[0] / self.hat_img.shape[1])
            hat_resized = cv2.resize(self.hat_img, (hat_width, hat_height))
            hat_x = top_head[0] - hat_width // 2
            hat_y = top_head[1] - hat_height
            image = self.overlay_image(image, hat_resized, hat_x, hat_y)
        return image

    def apply_mask_filter(self, image, face_landmarks):
        mouth_top = face_landmarks[13]
        mouth_bottom = face_landmarks[14]
        left_mouth = face_landmarks[78]
        right_mouth = face_landmarks[308]
        mouth_width = math.sqrt((left_mouth[0] - right_mouth[0])**2 + (left_mouth[1] - right_mouth[1])**2)
        mouth_height = math.sqrt((mouth_top[0] - mouth_bottom[0])**2 + (mouth_top[1] - mouth_bottom[1])**2)
        if self.mask_img is not None:
            mask_width = int(mouth_width * 2.5)
            mask_height = int(mask_width * self.mask_img.shape[0] / self.mask_img.shape[1])
            mask_resized = cv2.resize(self.mask_img, (mask_width, mask_height))
            mask_x = left_mouth[0] - int(mask_width * 0.2)
            mask_y = mouth_top[1] - int(mask_height * 0.4)
            image = self.overlay_image(image, mask_resized, mask_x, mask_y)
        return image

    # ============================================================
    # Integração do Pac-Man no modo "game"
    # ============================================================
    def init_game(self):
        self.pacman_game = PacManGame(self.width, self.height)
        self.pacman_last_update = time.time()

    def process_game(self, image, hand_landmarks, face_landmarks):
        if self.pacman_game is None:
            return image
        # Atualiza a direção do Pac-Man com base na posição da mão
        if hand_landmarks and len(hand_landmarks) > 0:
            index_tip = hand_landmarks[0][8]
            hand_x, hand_y = index_tip[0], index_tip[1]
            cv2.circle(image, (hand_x, hand_y), 10, (0, 255, 0), -1)
            self.pacman_game.update_pacman_direction(hand_x, hand_y)
        # Atualiza o jogo a cada 150ms
        current_time = time.time()
        if not hasattr(self, 'pacman_last_update'):
            self.pacman_last_update = current_time
        if current_time - self.pacman_last_update > 0.15:
            self.pacman_game.update()
            self.pacman_last_update = current_time
        image = self.pacman_game.draw(image)
        return image

    def run(self):
        try:
            self.mode = "menu"
            while self.cap.isOpened():
                success, image = self.cap.read()
                if not success:
                    print("Falha ao capturar frame da webcam.")
                    break
                display_image = self.process_frame(image)
                cv2.imshow('AR Virtual Assistant', display_image)
                key = cv2.waitKey(5) & 0xFF
                if key == 27:
                    break
                elif key == ord('m'):
                    self.mode = "menu"
                elif key == ord('d'):
                    self.mode = "drawing"
                elif key == ord('f'):
                    self.mode = "filters"
                    self.current_filter = self.filters[self.filter_index]
                elif key == ord('g'):
                    self.mode = "game"
                    self.init_game()
                elif key == ord('c') and self.mode == "drawing":
                    self.drawings = []
                    self.current_drawing = deque(maxlen=1000)
                elif key == ord('n') and self.mode == "filters":
                    self.filter_index = (self.filter_index + 1) % len(self.filters)
                    self.current_filter = self.filters[self.filter_index]
                elif key == ord('p') and self.mode == "filters":
                    self.filter_index = (self.filter_index - 1) % len(self.filters)
                    self.current_filter = self.filters[self.filter_index]
                # Reiniciar o jogo Pac-Man se pressionar 'r'
                elif key == ord('r') and self.mode == "game" and self.pacman_game is not None:
                    self.pacman_game.reset()
        finally:
            self.hands.close()
            self.face_mesh.close()
            self.cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        app = ARVirtualAssistant()
        app.run()
    except Exception as e:
        print(f"Erro na execução: {e}")
