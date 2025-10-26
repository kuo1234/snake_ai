"""
貪吃蛇AI可視化演示
觀看訓練好的AI如何遊戲
支援 Q-learning、PPO V1、PPO V2 和 PPO V3 (Curriculum Learning) 四種模型
"""

import sys
import time
import os
import glob
import pygame
import numpy as np
from snake_ai import SnakeAI
from envs.snake_game import SnakeGame

# 嘗試導入 PPO 相關模組
try:
    from stable_baselines3 import PPO
    from envs.gym_snake_env import GymSnakeEnv
    from envs.gym_snake_env_v2 import GymSnakeEnvV2
    HAS_PPO = True
except ImportError:
    HAS_PPO = False
    print("注意: stable_baselines3 未安裝，PPO 功能不可用")

# UI 顏色定義
COLOR_BG = (20, 20, 30)
COLOR_TITLE = (255, 255, 255)
COLOR_TEXT = (200, 200, 200)
COLOR_BUTTON = (50, 100, 150)
COLOR_BUTTON_HOVER = (70, 130, 180)
COLOR_BUTTON_SELECTED = (100, 180, 100)
COLOR_BORDER = (100, 100, 120)


def get_chinese_font(size):
    """獲取支持中文的字體"""
    # Windows 常見中文字體列表（按優先級排序）
    chinese_fonts = [
        'microsoftyahei',      # 微軟雅黑
        'microsoftyaheiui',    # 微軟雅黑 UI
        'simsun',              # 宋體
        'simhei',              # 黑體
        'kaiti',               # 楷體
        'fangsong',            # 仿宋
        'msgothic',            # MS Gothic (日文，但也支持中文)
        'arial',               # Arial Unicode MS
    ]
    
    # 嘗試加載中文字體
    for font_name in chinese_fonts:
        try:
            font = pygame.font.SysFont(font_name, size)
            # 測試是否能渲染中文
            test_surface = font.render('測試', True, (255, 255, 255))
            if test_surface.get_width() > 0:
                return font
        except:
            continue
    
    # 如果都失敗了，使用默認字體
    print("警告: 無法加載中文字體，可能會顯示亂碼")
    return pygame.font.Font(None, size)


def list_available_models():
    """掃描並列出所有可用的模型文件"""
    models = {
        'qlearning': [],
        'ppo_v1': [],
        'ppo_v2': [],
        'ppo_v3': []
    }
    
    # Q-learning 模型 (在根目錄)
    qlearning_patterns = ["snake_ai_*.pkl"]
    for pattern in qlearning_patterns:
        for f in glob.glob(pattern):
            if os.path.exists(f):
                models['qlearning'].append(f)
    
    # PPO V1 模型
    ppo_v1_dir = "models/ppo_snake"
    if os.path.exists(ppo_v1_dir):
        # 最佳模型
        best = os.path.join(ppo_v1_dir, "best_model", "best_model.zip")
        if os.path.exists(best):
            models['ppo_v1'].append(best)
        # 所有 checkpoint
        for f in glob.glob(os.path.join(ppo_v1_dir, "*.zip")):
            models['ppo_v1'].append(f)
    
    # PPO V2 模型
    ppo_v2_dir = "models/ppo_snake_v2"
    if os.path.exists(ppo_v2_dir):
        # 最佳模型
        best = os.path.join(ppo_v2_dir, "best_model", "best_model.zip")
        if os.path.exists(best):
            models['ppo_v2'].append(best)
        # 所有 checkpoint
        for f in glob.glob(os.path.join(ppo_v2_dir, "*.zip")):
            models['ppo_v2'].append(f)
    
    # PPO V3 模型（課程學習，4個階段）
    ppo_v3_base = "models/ppo_snake_v3_curriculum"
    if os.path.exists(ppo_v3_base):
        stages = ["Stage1_Novice", "Stage2_Intermediate", "Stage3_Advanced", "Stage4_Master"]
        for stage in stages:
            stage_dir = os.path.join(ppo_v3_base, stage)
            if os.path.exists(stage_dir):
                # 最佳模型（畢業模型）
                best = os.path.join(stage_dir, "best_model", "best_model.zip")
                if os.path.exists(best):
                    models['ppo_v3'].append(best)
                # 最終模型
                final = os.path.join(stage_dir, "model.zip")
                if os.path.exists(final):
                    models['ppo_v3'].append(final)
                # Checkpoints
                checkpoint_dir = os.path.join(stage_dir, "checkpoints")
                if os.path.exists(checkpoint_dir):
                    for f in glob.glob(os.path.join(checkpoint_dir, "*.zip")):
                        models['ppo_v3'].append(f)
    
    return models


class UIButton:
    """pygame UI 按鈕類"""
    def __init__(self, x, y, width, height, text, value=None):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.value = value if value is not None else text
        self.is_hovered = False
        self.is_selected = False
    
    def draw(self, screen, font):
        # 選擇顏色
        if self.is_selected:
            color = COLOR_BUTTON_SELECTED
        elif self.is_hovered:
            color = COLOR_BUTTON_HOVER
        else:
            color = COLOR_BUTTON
        
        # 繪製按鈕
        pygame.draw.rect(screen, color, self.rect, border_radius=8)
        pygame.draw.rect(screen, COLOR_BORDER, self.rect, 2, border_radius=8)
        
        # 繪製文字
        text_surface = font.render(self.text, True, COLOR_TITLE)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)
    
    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.is_hovered:
                return True
        return False


class ModelSelector:
    """模型選擇器 UI"""
    def __init__(self, width=800, height=600):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("選擇 AI 模型和設定")
        
        # 使用支持中文的字體
        self.font_large = get_chinese_font(48)
        self.font_medium = get_chinese_font(36)
        self.font_small = get_chinese_font(24)
        
        self.selected_model_type = None
        self.selected_model_path = None
        self.selected_board_size = None
        
        self.models = list_available_models()
        self.buttons = {}
        self.setup_ui()
    
    def setup_ui(self):
        """設置 UI 元素"""
        # 模型類型選擇按鈕 (第一頁)
        button_width = 250
        button_height = 80
        start_y = 180
        spacing = 100
        center_x = self.width // 2 - button_width // 2
        
        self.type_buttons = []
        
        if self.models['qlearning']:
            btn = UIButton(center_x, start_y, button_width, button_height, 
                          f"Q-Learning ({len(self.models['qlearning'])}個)", "qlearning")
            self.type_buttons.append(btn)
        
        if HAS_PPO and self.models['ppo_v1']:
            btn = UIButton(center_x, start_y + spacing, button_width, button_height, 
                          f"PPO V1 ({len(self.models['ppo_v1'])}個)", "ppo_v1")
            self.type_buttons.append(btn)
        
        if HAS_PPO and self.models['ppo_v2']:
            btn = UIButton(center_x, start_y + spacing * 2, button_width, button_height, 
                          f"PPO V2 ({len(self.models['ppo_v2'])}個)", "ppo_v2")
            self.type_buttons.append(btn)
        
        if HAS_PPO and self.models['ppo_v3']:
            btn = UIButton(center_x, start_y + spacing * 3, button_width, button_height, 
                          f"PPO V3 🎓 ({len(self.models['ppo_v3'])}個)", "ppo_v3")
            self.type_buttons.append(btn)
        
        # 輸入框相關（自定義棋盤大小）
        self.input_active = True  # 直接進入輸入模式
        self.input_text = ""
        self.input_rect = pygame.Rect(center_x, 250, button_width, 60)
    
    def draw_text(self, text, y, font=None, color=COLOR_TEXT):
        """在屏幕中央繪製文字"""
        if font is None:
            font = self.font_medium
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect(center=(self.width // 2, y))
        self.screen.blit(text_surface, text_rect)
    
    def draw_model_type_selection(self):
        """繪製模型類型選擇頁面"""
        self.screen.fill(COLOR_BG)
        
        # 標題
        self.draw_text("選擇 AI 模型類型", 80, self.font_large, COLOR_TITLE)
        
        # 按鈕
        for btn in self.type_buttons:
            btn.draw(self.screen, self.font_medium)
        
        # 提示
        self.draw_text("使用滑鼠點擊選擇", self.height - 50, self.font_small)
        
        pygame.display.flip()
    
    def draw_model_file_selection(self):
        """繪製具體模型文件選擇頁面"""
        self.screen.fill(COLOR_BG)
        
        # 標題
        type_names = {
            'qlearning': 'Q-Learning',
            'ppo_v1': 'PPO V1',
            'ppo_v2': 'PPO V2',
            'ppo_v3': 'PPO V3 (課程學習)'
        }
        title = f"選擇 {type_names.get(self.selected_model_type, '')} 模型"
        self.draw_text(title, 60, self.font_large, COLOR_TITLE)
        
        # 模型文件按鈕
        models = self.models[self.selected_model_type]
        button_width = 600
        button_height = 60
        start_y = 150
        center_x = self.width // 2 - button_width // 2
        
        # 創建臨時按鈕
        if not hasattr(self, 'model_file_buttons'):
            self.model_file_buttons = []
            for i, model_path in enumerate(models):
                # 簡化顯示名稱
                display_name = os.path.basename(model_path)
                
                # 特殊標記
                if 'best_model' in model_path:
                    display_name = "★ " + display_name + " (最佳)"
                
                # V3 階段標記
                if self.selected_model_type == 'ppo_v3':
                    if 'Stage1_Novice' in model_path:
                        display_name = "🎓 階段1: " + display_name + " (6x6)"
                    elif 'Stage2_Intermediate' in model_path:
                        display_name = "🎓 階段2: " + display_name + " (8x8)"
                    elif 'Stage3_Advanced' in model_path:
                        display_name = "🎓 階段3: " + display_name + " (10x10)"
                    elif 'Stage4_Master' in model_path:
                        display_name = "🎓 階段4: " + display_name + " (12x12)"
                
                btn = UIButton(center_x, start_y + i * 70, button_width, button_height, 
                              display_name[:70], model_path)
                self.model_file_buttons.append(btn)
        
        # 繪製按鈕
        for btn in self.model_file_buttons:
            btn.draw(self.screen, self.font_small)
        
        # 返回提示
        self.draw_text("按 ESC 返回上一步", self.height - 50, self.font_small)
        
        pygame.display.flip()
    
    def draw_board_size_selection(self):
        """繪製板子大小選擇頁面"""
        self.screen.fill(COLOR_BG)
        
        # 標題
        self.draw_text("輸入棋盤大小", 80, self.font_large, COLOR_TITLE)
        
        # 顯示建議範圍
        self.draw_text("建議範圍: 5-20", 150, self.font_small, COLOR_TEXT)
        self.draw_text("(6=簡單, 8=標準, 10=困難, 12=極難)", 180, self.font_small, COLOR_TEXT)
        
        # 繪製輸入框
        color = COLOR_BUTTON_HOVER
        pygame.draw.rect(self.screen, color, self.input_rect, border_radius=8)
        pygame.draw.rect(self.screen, COLOR_BORDER, self.input_rect, 3, border_radius=8)
        
        # 繪製輸入的文字
        input_display = self.input_text + "|" if self.input_text or self.input_active else "請輸入數字..."
        text_color = COLOR_TITLE if self.input_text else (150, 150, 150)
        text_surface = self.font_large.render(input_display, True, text_color)
        text_rect = text_surface.get_rect(center=self.input_rect.center)
        self.screen.blit(text_surface, text_rect)
        
        # 提示按 Enter 確認
        self.draw_text("按 Enter 確認", self.input_rect.y + 100, self.font_medium, COLOR_TEXT)
        self.draw_text("按 ESC 返回上一步", self.height - 50, self.font_small)
        
        pygame.display.flip()
    
    def run(self):
        """運行選擇器"""
        clock = pygame.time.Clock()
        stage = 'type'  # type -> file -> board -> done
        
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return None, None, None
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        if stage == 'file':
                            stage = 'type'
                            self.model_file_buttons = []
                        elif stage == 'board':
                            stage = 'file'
                            self.input_text = ""
                        elif stage == 'type':
                            pygame.quit()
                            return None, None, None
                    
                    # 處理自定義大小輸入
                    elif stage == 'board':
                        if event.key == pygame.K_RETURN:
                            # 確認輸入
                            if self.input_text:
                                try:
                                    size = int(self.input_text)
                                    if 5 <= size <= 20:
                                        self.selected_board_size = size
                                        pygame.quit()
                                        return self.selected_model_type, self.selected_model_path, self.selected_board_size
                                    else:
                                        print(f"警告: 棋盤大小必須在 5-20 之間，您輸入的是 {size}")
                                        self.input_text = ""
                                except ValueError:
                                    print(f"警告: 請輸入有效的數字")
                                    self.input_text = ""
                        elif event.key == pygame.K_BACKSPACE:
                            self.input_text = self.input_text[:-1]
                        elif event.unicode.isdigit() and len(self.input_text) < 2:
                            self.input_text += event.unicode
                
                # 處理不同階段的事件
                if stage == 'type':
                    for btn in self.type_buttons:
                        btn.handle_event(event)
                        if event.type == pygame.MOUSEBUTTONDOWN and btn.is_hovered:
                            self.selected_model_type = btn.value
                            stage = 'file'
                            break
                
                elif stage == 'file':
                    for btn in self.model_file_buttons:
                        btn.handle_event(event)
                        if event.type == pygame.MOUSEBUTTONDOWN and btn.is_hovered:
                            self.selected_model_path = btn.value
                            stage = 'board'
                            break
                
                # board 階段不需要處理滑鼠點擊（只用鍵盤輸入）
            
            # 繪製當前階段
            if stage == 'type':
                self.draw_model_type_selection()
            elif stage == 'file':
                self.draw_model_file_selection()
            elif stage == 'board':
                self.draw_board_size_selection()
            
            clock.tick(60)
        
        pygame.quit()
        return None, None, None



def get_user_choice_board_size():
    """讓用戶選擇棋盤大小"""
    print("\n請選擇棋盤大小:")
    print("  1. 6x6 (簡單)")
    print("  2. 8x8 (標準) 推薦")
    print("  3. 10x10 (困難)")
    print("  4. 12x12 (極難)")
    
    while True:
        try:
            choice = input("\n請輸入選擇 (1-4，直接按Enter選擇8x8): ").strip()
            
            if choice == '' or choice == '2':
                return 8
            elif choice == '1':
                return 6
            elif choice == '3':
                return 10
            elif choice == '4':
                return 12
            else:
                print("無效選擇，請重新輸入")
        except ValueError:
            print("無效輸入，請輸入數字")

def watch_ai_play(board_size=None, model_path=None):
    """觀看AI遊戲並顯示決策過程"""
    
    # 讓用戶選擇棋盤大小
    if board_size is None:
        board_size = get_user_choice_board_size()
    
    # 載入訓練好的AI
    ai = SnakeAI(board_size=board_size)
    
    # 嘗試載入最佳模型
    if model_path:
        models_to_try = [model_path]
    else:
        models_to_try = [
            "snake_ai_standard.pkl",
            "snake_ai_quick.pkl", 
            "snake_ai_final.pkl"
        ]
    
    model_loaded = False
    for model_file in models_to_try:
        try:
            ai.load_model(model_file)
            print(f"成功載入模型: {model_file}")
            model_loaded = True
            break
        except:
            continue
    
    if not model_loaded:
        print("警告: 未找到預訓練模型，使用隨機AI")
    
    # 創建遊戲實例
    game = SnakeGame(silent_mode=False, board_size=ai.board_size)
    
    action_names = {0: "UP", 1: "LEFT", 2: "RIGHT", 3: "DOWN"}
    
    print("\n" + "="*60)
    print(f"=== AI貪吃蛇演示 (棋盤: {board_size}x{board_size}) ===")
    print("="*60)
    print("觀看AI如何使用貝爾曼方程學到的策略")
    print("按ESC鍵或關閉窗口結束演示")
    print("按空格鍵暫停/繼續")
    print("="*60)
    
    try:
        game_count = 1
        
        while True:
            print(f"\n--- 遊戲 {game_count} 開始 ---")
            
            # 重置遊戲
            game.reset()
            state = ai.get_state(game)
            steps = 0
            max_steps = ai.board_size * ai.board_size * 3
            
            # 遊戲循環
            while steps < max_steps:
                # AI選擇動作
                action = ai.choose_action(state, training=False)
                q_values = ai.get_q_values(state)
                
                # 顯示AI的思考過程
                print(f"步驟 {steps+1:3d}: 動作={action_names[action]:5s} "
                      f"Q值=[{q_values[0]:6.2f}, {q_values[1]:6.2f}, {q_values[2]:6.2f}, {q_values[3]:6.2f}] "
                      f"分數={game.score:3d}")
                
                # 執行動作
                done, info = game.step(action)
                
                # 渲染遊戲
                if hasattr(game, 'render') and game.screen:
                    game.render()
                
                # 檢查pygame事件
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            pygame.quit()
                            return
                        elif event.key == pygame.K_SPACE:
                            # 空格鍵暫停/繼續
                            input("按Enter繼續...")
                
                if done:
                    print(f"遊戲結束! 最終分數: {game.score}, 總步數: {steps+1}")
                    
                    # 等待一下再開始下一局
                    time.sleep(2)
                    break
                
                # 更新狀態
                state = ai.get_state(game)
                steps += 1
                
                # 控制遊戲速度
                time.sleep(0.2)
            
            game_count += 1
            
            if game_count > 5:  # 最多觀看5局
                print("演示結束")
                break
                
    except KeyboardInterrupt:
        print("\n演示被用戶中斷")
    
    finally:
        if pygame.get_init():
            pygame.quit()

def compare_before_after_training():
    """比較訓練前後的AI表現"""
    
    print("=== 訓練前後比較 ===")
    
    # 未訓練的AI
    untrained_ai = SnakeAI(board_size=8)
    
    # 訓練好的AI
    trained_ai = SnakeAI(board_size=8)
    try:
        trained_ai.load_model("snake_ai_standard.pkl")
        print("載入訓練好的AI模型")
    except:
        print("警告: 未找到訓練好的模型")
        return
    
    # 測試兩個AI
    def test_ai_performance(ai, name, num_games=5):
        scores = []
        print(f"\n測試 {name}:")
        
        for i in range(num_games):
            score, steps = ai.play_game(render=False)
            scores.append(score)
            print(f"  遊戲 {i+1}: 分數={score}, 步數={steps}")
        
        avg_score = sum(scores) / len(scores)
        print(f"  平均分數: {avg_score:.2f}")
        return scores
    
    untrained_scores = test_ai_performance(untrained_ai, "未訓練AI")
    trained_scores = test_ai_performance(trained_ai, "訓練後AI")
    
    print(f"\n=== 比較結果 ===")
    print(f"未訓練AI平均分數: {sum(untrained_scores)/len(untrained_scores):.2f}")
    print(f"訓練後AI平均分數: {sum(trained_scores)/len(trained_scores):.2f}")
    print(f"改進倍數: {(sum(trained_scores)/len(trained_scores)) / max(1, sum(untrained_scores)/len(untrained_scores)):.2f}x")

def analyze_ai_strategy():
    """分析AI學到的策略"""
    
    ai = SnakeAI(board_size=8)
    try:
        ai.load_model("snake_ai_standard.pkl")
    except:
        print("需要先訓練AI模型")
        return
    
    print("=== AI策略分析 ===")
    print(f"Q表大小: {len(ai.q_table)} 個狀態")
    
    # 分析Q值分布
    all_q_values = []
    for state_q_values in ai.q_table.values():
        all_q_values.extend(state_q_values)
    
    import numpy as np
    print(f"Q值統計:")
    print(f"  最大值: {max(all_q_values):.2f}")
    print(f"  最小值: {min(all_q_values):.2f}")
    print(f"  平均值: {np.mean(all_q_values):.2f}")
    print(f"  標準差: {np.std(all_q_values):.2f}")
    
    # 找出最重要的狀態
    print(f"\n最有價值的狀態-動作對:")
    state_action_values = []
    action_names = {0: "UP", 1: "LEFT", 2: "RIGHT", 3: "DOWN"}
    
    for state, q_values in ai.q_table.items():
        for action, q_value in enumerate(q_values):
            state_action_values.append((q_value, state, action))
    
    state_action_values.sort(reverse=True)
    
    for i, (q_value, state, action) in enumerate(state_action_values[:5]):
        print(f"  {i+1}. Q值={q_value:.2f}, 動作={action_names[action]}, 狀態={state}")


def watch_ppo_play(version='v1', board_size=None, model_path=None):
    """觀看 PPO AI 遊戲並顯示決策過程
    
    Args:
        version: 'v1', 'v2', 或 'v3'，選擇 PPO 版本
        board_size: 棋盤大小，None 表示讓用戶選擇
        model_path: 模型路徑，None 表示自動搜尋
    """
    
    if not HAS_PPO:
        print("錯誤: 需要安裝 stable_baselines3 才能使用 PPO 模型")
        print("請執行: pip install stable-baselines3 torch")
        return
    
    # 讓用戶選擇棋盤大小
    if board_size is None:
        board_size = get_user_choice_board_size()
    
    # 嘗試載入 PPO 模型
    if model_path:
        models_to_try = [model_path]
    elif version == 'v3':
        # V3 根據棋盤大小選擇對應的階段模型
        stage_map = {
            6: "Stage1_Novice",
            8: "Stage2_Intermediate",
            10: "Stage3_Advanced",
            12: "Stage4_Master"
        }
        stage = stage_map.get(board_size, "Stage2_Intermediate")
        models_to_try = [
            f"models/ppo_snake_v3_curriculum/{stage}/best_model/best_model.zip",
            f"models/ppo_snake_v3_curriculum/{stage}/model.zip",
        ]
    elif version == 'v2':
        models_to_try = [
            "models/ppo_snake_v2/best_model/best_model.zip",
            "models/ppo_snake_v2/ppo_snake_v2_final.zip",
        ]
    else:  # v1
        models_to_try = [
            "models/ppo_snake/best_model/best_model.zip",
            "models/ppo_snake/ppo_snake_final.zip",
        ]
    
    model = None
    model_path_loaded = None
    for path in models_to_try:
        if os.path.exists(path):
            try:
                model = PPO.load(path)
                model_path_loaded = path
                print(f"成功載入 PPO {version.upper()} 模型: {path}")
                break
            except Exception as e:
                print(f"載入 {path} 失敗: {e}")
                continue
    
    if model is None:
        print(f"警告: 未找到 PPO {version.upper()} 預訓練模型")
        print(f"請先訓練模型:")
        if version == 'v3':
            print(f"  python snake_ai_ppo_v3.py --mode train")
        elif version == 'v2':
            print(f"  python snake_ai_ppo_v2.py --mode train --timesteps 500000 --board-size {board_size}")
        else:
            print(f"  python snake_ai_ppo.py --mode train --timesteps 100000 --board-size {board_size}")
        return
    
    # 創建遊戲環境
    if version == 'v3':
        env = GymSnakeEnvV2(board_size=board_size, render_mode="human")
        version_name = "PPO V3 (Curriculum Learning)"
        features = [
            "課程學習：循序漸進訓練",
            "遷移學習：知識逐級傳遞",
            "更大神經網路 [256, 256, 128]",
            "16-d 觀察空間 + V2 獎勵塑形",
        ]
    elif version == 'v2':
        env = GymSnakeEnvV2(board_size=board_size, render_mode="human")
        version_name = "PPO V2 (Enhanced Collision Avoidance)"
        features = [
            "使用深度神經網路學習策略",
            "漸進式碰撞懲罰 (頸部 -50)",
            "16-d 觀察空間 (含身體距離感知)",
            "困境偵測與避免",
        ]
    else:
        env = GymSnakeEnv(board_size=board_size, render_mode="human")
        version_name = "PPO V1 (Basic)"
        features = [
            "使用深度神經網路學習策略",
            "穩定且高效的訓練過程",
            "12-d 觀察空間",
            "基礎獎勵塑形",
        ]
    
    action_names = {0: "UP", 1: "LEFT", 2: "RIGHT", 3: "DOWN"}
    
    print("\n" + "="*70)
    print(f"{version_name} AI 演示")
    print(f"棋盤大小: {board_size}x{board_size}")
    print("="*70)
    print("特點:")
    for feature in features:
        print(f"  • {feature}")
    print("="*70)
    print("操作說明:")
    print("  • 按 ESC 鍵或關閉窗口結束演示")
    print("  • 按 空格鍵 暫停/繼續")
    print("="*70)
    
    try:
        game_count = 1
        
        while True:
            print(f"\n--- 遊戲 {game_count} 開始 ---")
            
            # 重置環境
            obs, info = env.reset()
            done = False
            steps = 0
            total_reward = 0
            
            # 遊戲循環
            while not done:
                # PPO 選擇動作 (deterministic=True 表示使用確定性策略)
                action, _states = model.predict(obs, deterministic=True)
                action = int(action)
                
                # 顯示 AI 的決策
                print(f"步驟 {steps+1:3d}: 動作={action_names[action]:5s} "
                      f"分數={info.get('score', 0):3d} "
                      f"蛇長={info.get('snake_length', 0):3d}")
                
                # 執行動作
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                done = terminated or truncated
                
                # 渲染遊戲
                env.render()
                
                # 檢查 pygame 事件
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        env.close()
                        return
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            pygame.quit()
                            env.close()
                            return
                        elif event.key == pygame.K_SPACE:
                            # 空格鍵暫停/繼續
                            input("按 Enter 繼續...")
                
                steps += 1
                
                # 控制遊戲速度
                time.sleep(0.15)
            
            # 遊戲結束統計
            final_score = info.get('score', 0)
            final_length = info.get('snake_length', 0)
            near_misses = info.get('near_miss_count', 0) if version == 'v2' else 'N/A'
            
            print(f"\n遊戲結束統計:")
            print(f"  最終分數: {final_score}")
            print(f"  蛇的長度: {final_length}")
            print(f"  總步數: {steps}")
            print(f"  總獎勵: {total_reward:.2f}")
            print(f"  成功率: {final_score}/{board_size**2-1} ({final_score/(board_size**2-1)*100:.1f}%)")
            if version == 'v2':
                print(f"  Near-miss 次數: {near_misses}")
            
            # 等待一下再開始下一局
            time.sleep(2)
            
            game_count += 1
            
            if game_count > 5:  # 最多觀看 5 局
                print("\n演示結束")
                break
                
    except KeyboardInterrupt:
        print("\n演示被用戶中斷")
    
    finally:
        env.close()
        if pygame.get_init():
            pygame.quit()


def compare_qlearning_vs_ppo():
    """比較 Q-learning 和 PPO 兩種方法的表現"""
    
    print("\n" + "="*60)
    print("Q-learning vs PPO 性能比較")
    print("="*60)
    
    # 測試 Q-learning
    print("\n測試 Q-learning AI...")
    qlearning_ai = SnakeAI(board_size=8)
    try:
        qlearning_ai.load_model("snake_ai_standard.pkl")
        print("✓ 載入 Q-learning 模型成功")
        
        qlearning_scores = []
        for i in range(10):
            score, steps = qlearning_ai.play_game(render=False)
            qlearning_scores.append(score)
            print(f"  遊戲 {i+1}: 分數={score}, 步數={steps}")
        
        qlearning_avg = np.mean(qlearning_scores)
        qlearning_max = max(qlearning_scores)
        print(f"  平均分數: {qlearning_avg:.2f}")
        print(f"  最高分數: {qlearning_max}")
        
    except Exception as e:
        print(f"✗ Q-learning 模型載入失敗: {e}")
        qlearning_avg = 0
        qlearning_max = 0
        qlearning_scores = []
    
    # 測試 PPO
    if HAS_PPO:
        print("\n測試 PPO AI...")
        
        models_to_try = [
            "models/ppo_snake/best_model/best_model.zip",
            "models/ppo_snake/ppo_snake_final.zip",
        ]
        
        ppo_model = None
        for path in models_to_try:
            if os.path.exists(path):
                try:
                    ppo_model = PPO.load(path)
                    print(f"✓ 載入 PPO 模型成功: {path}")
                    break
                except:
                    continue
        
        if ppo_model is not None:
            env = GymSnakeEnv(board_size=8, render_mode=None)
            ppo_scores = []
            
            for i in range(10):
                obs, info = env.reset()
                done = False
                steps = 0
                
                while not done:
                    action, _ = ppo_model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    steps += 1
                
                score = info.get('score', 0)
                ppo_scores.append(score)
                print(f"  遊戲 {i+1}: 分數={score}, 步數={steps}")
            
            ppo_avg = np.mean(ppo_scores)
            ppo_max = max(ppo_scores)
            print(f"  平均分數: {ppo_avg:.2f}")
            print(f"  最高分數: {ppo_max}")
            
            env.close()
        else:
            print("✗ PPO 模型載入失敗")
            ppo_avg = 0
            ppo_max = 0
            ppo_scores = []
    else:
        print("\n✗ PPO 不可用（需要安裝 stable_baselines3）")
        ppo_avg = 0
        ppo_max = 0
        ppo_scores = []
    
    # 顯示比較結果
    print("\n" + "="*60)
    print("比較結果總結")
    print("="*60)
    
    print(f"\n{'方法':<15} {'平均分數':<12} {'最高分數':<12} {'優勢'}")
    print("-" * 60)
    print(f"{'Q-learning':<15} {qlearning_avg:<12.2f} {qlearning_max:<12} 傳統方法，穩定")
    print(f"{'PPO':<15} {ppo_avg:<12.2f} {ppo_max:<12} 深度學習，擴展性強")
    
    if ppo_avg > 0 and qlearning_avg > 0:
        improvement = (ppo_avg - qlearning_avg) / qlearning_avg * 100
        print(f"\nPPO 相比 Q-learning 的改進: {improvement:+.1f}%")
    
    print("\n" + "="*60)
    print("算法特性比較:")
    print("="*60)
    print("\nQ-learning:")
    print("  ✓ 簡單易懂，實現簡單")
    print("  ✓ 訓練快速（小棋盤）")
    print("  ✗ 狀態空間爆炸（大棋盤）")
    print("  ✗ 難以擴展到複雜問題")
    
    print("\nPPO:")
    print("  ✓ 使用深度神經網路，擴展性強")
    print("  ✓ 可處理大規模狀態空間")
    print("  ✓ 支援多進程並行訓練")
    print("  ✓ 現代 RL 標準方法")
    print("  ✗ 需要更多訓練時間")
    print("  ✗ 需要調整超參數")
    print("="*60)

def compare_all_methods():
    """比較 Q-learning、PPO V1 和 PPO V2 三種方法的表現"""
    
    print("\n" + "="*70)
    print("Q-learning vs PPO V1 vs PPO V2 性能比較")
    print("="*70)
    
    # 讓用戶選擇棋盤大小
    board_size = get_user_choice_board_size()
    print(f"\n使用棋盤大小: {board_size}x{board_size}")
    
    num_games = 10
    results = {}
    
    # 測試 Q-learning
    print("\n" + "─"*70)
    print("測試 Q-learning AI...")
    print("─"*70)
    qlearning_ai = SnakeAI(board_size=board_size)
    try:
        qlearning_ai.load_model("snake_ai_standard.pkl")
        print("✓ 載入 Q-learning 模型成功")
        
        qlearning_scores = []
        for i in range(num_games):
            score, steps = qlearning_ai.play_game(render=False)
            qlearning_scores.append(score)
            print(f"  遊戲 {i+1:2d}: 分數={score:3d}, 步數={steps:4d}")
        
        results['Q-learning'] = {
            'scores': qlearning_scores,
            'avg': np.mean(qlearning_scores),
            'max': max(qlearning_scores),
            'min': min(qlearning_scores),
            'std': np.std(qlearning_scores)
        }
        print(f"  平均分數: {results['Q-learning']['avg']:.2f} ± {results['Q-learning']['std']:.2f}")
        
    except Exception as e:
        print(f"✗ Q-learning 模型載入失敗: {e}")
        results['Q-learning'] = None
    
    # 測試 PPO V1
    print("\n" + "─"*70)
    print("測試 PPO V1 AI...")
    print("─"*70)
    
    ppo_v1_models = [
        "models/ppo_snake/best_model/best_model.zip",
        "models/ppo_snake/ppo_snake_final.zip",
    ]
    
    ppo_v1_model = None
    for path in ppo_v1_models:
        if os.path.exists(path):
            try:
                ppo_v1_model = PPO.load(path)
                print(f"✓ 載入 PPO V1 模型成功: {path}")
                break
            except:
                continue
    
    if ppo_v1_model is not None:
        env = GymSnakeEnv(board_size=board_size, render_mode=None)
        ppo_v1_scores = []
        
        for i in range(num_games):
            obs, info = env.reset()
            done = False
            steps = 0
            
            while not done:
                action, _ = ppo_v1_model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                steps += 1
            
            score = info.get('score', 0)
            ppo_v1_scores.append(score)
            print(f"  遊戲 {i+1:2d}: 分數={score:3d}, 步數={steps:4d}")
        
        results['PPO V1'] = {
            'scores': ppo_v1_scores,
            'avg': np.mean(ppo_v1_scores),
            'max': max(ppo_v1_scores),
            'min': min(ppo_v1_scores),
            'std': np.std(ppo_v1_scores)
        }
        print(f"  平均分數: {results['PPO V1']['avg']:.2f} ± {results['PPO V1']['std']:.2f}")
        
        env.close()
    else:
        print("✗ PPO V1 模型載入失敗")
        results['PPO V1'] = None
    
    # 測試 PPO V2
    print("\n" + "─"*70)
    print("測試 PPO V2 AI (Enhanced Collision Avoidance)...")
    print("─"*70)
    
    ppo_v2_models = [
        "models/ppo_snake_v2/best_model/best_model.zip",
        "models/ppo_snake_v2/ppo_snake_v2_final.zip",
    ]
    
    ppo_v2_model = None
    for path in ppo_v2_models:
        if os.path.exists(path):
            try:
                ppo_v2_model = PPO.load(path)
                print(f"✓ 載入 PPO V2 模型成功: {path}")
                break
            except:
                continue
    
    if ppo_v2_model is not None:
        env = GymSnakeEnvV2(board_size=board_size, render_mode=None)
        ppo_v2_scores = []
        ppo_v2_near_misses = []
        
        for i in range(num_games):
            obs, info = env.reset()
            done = False
            steps = 0
            
            while not done:
                action, _ = ppo_v2_model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                steps += 1
            
            score = info.get('score', 0)
            near_miss = info.get('near_miss_count', 0)
            ppo_v2_scores.append(score)
            ppo_v2_near_misses.append(near_miss)
            print(f"  遊戲 {i+1:2d}: 分數={score:3d}, 步數={steps:4d}, Near-miss={near_miss:2d}")
        
        results['PPO V2'] = {
            'scores': ppo_v2_scores,
            'avg': np.mean(ppo_v2_scores),
            'max': max(ppo_v2_scores),
            'min': min(ppo_v2_scores),
            'std': np.std(ppo_v2_scores),
            'near_misses': np.mean(ppo_v2_near_misses)
        }
        print(f"  平均分數: {results['PPO V2']['avg']:.2f} ± {results['PPO V2']['std']:.2f}")
        print(f"  平均 Near-miss: {results['PPO V2']['near_misses']:.1f}")
        
        env.close()
    else:
        print("✗ PPO V2 模型載入失敗")
        results['PPO V2'] = None
    
    # 顯示比較結果
    print("\n" + "="*70)
    print("比較結果總結")
    print("="*70)
    
    print(f"\n{'方法':<15} {'平均分數':<15} {'最高分':<10} {'最低分':<10} {'標準差':<10}")
    print("─" * 70)
    
    for method, data in results.items():
        if data:
            print(f"{method:<15} {data['avg']:<15.2f} {data['max']:<10} {data['min']:<10} {data['std']:<10.2f}")
        else:
            print(f"{method:<15} {'N/A':<15} {'N/A':<10} {'N/A':<10} {'N/A':<10}")
    
    # 計算改進百分比
    print("\n" + "="*70)
    print("改進分析")
    print("="*70)
    
    if results.get('Q-learning') and results.get('PPO V1'):
        improvement = (results['PPO V1']['avg'] - results['Q-learning']['avg']) / max(results['Q-learning']['avg'], 0.01) * 100
        print(f"\nPPO V1 相比 Q-learning: {improvement:+.1f}%")
    
    if results.get('Q-learning') and results.get('PPO V2'):
        improvement = (results['PPO V2']['avg'] - results['Q-learning']['avg']) / max(results['Q-learning']['avg'], 0.01) * 100
        print(f"PPO V2 相比 Q-learning: {improvement:+.1f}%")
    
    if results.get('PPO V1') and results.get('PPO V2'):
        improvement = (results['PPO V2']['avg'] - results['PPO V1']['avg']) / max(results['PPO V1']['avg'], 0.01) * 100
        print(f"PPO V2 相比 PPO V1: {improvement:+.1f}%")
    
    print("\n" + "="*70)
    print("特性對比")
    print("="*70)
    
    print("\nQ-learning:")
    print("  ✓ 簡單易懂")
    print("  ✓ 訓練快速（小棋盤）")
    print("  ✗ 狀態空間爆炸（大棋盤）")
    
    print("\nPPO V1:")
    print("  ✓ 深度神經網路")
    print("  ✓ 擴展性強")
    print("  ✓ 12-d 觀察空間")
    print("  ✗ 容易撞到自己")
    
    print("\nPPO V2:")
    print("  ✓ 漸進式碰撞懲罰")
    print("  ✓ 16-d 觀察空間 (含身體感知)")
    print("  ✓ 困境偵測")
    print("  ✓ 更好的避障能力")
    
    print("="*70)


def advanced_mode():
    """進階模式：讓用戶選擇特定模型和棋盤大小"""
    
    print("\n" + "="*70)
    print("進階模式 - 自訂選擇")
    print("="*70)
    
    # 顯示所有可用模型
    available_models = list_available_models()
    
    print("\n請選擇 AI 類型:")
    print("  1. Q-learning")
    print("  2. PPO V1")
    print("  3. PPO V2 (推薦)")
    
    ai_choice = input("\n請輸入選擇 (1-3): ").strip()
    
    if ai_choice == '1':
        # Q-learning
        if not available_models['qlearning']:
            print("\n沒有可用的 Q-learning 模型")
            return
        
        print("\n可用的 Q-learning 模型:")
        for i, model in enumerate(available_models['qlearning'], 1):
            print(f"  {i}. {model}")
        
        if len(available_models['qlearning']) == 1:
            model_idx = 0
        else:
            model_choice = input(f"\n請選擇模型 (1-{len(available_models['qlearning'])}): ").strip()
            model_idx = int(model_choice) - 1
        
        model_path = available_models['qlearning'][model_idx]
        board_size = get_user_choice_board_size()
        
        print(f"\n載入模型: {model_path}")
        print(f"棋盤大小: {board_size}x{board_size}")
        
        watch_ai_play(board_size=board_size, model_path=model_path)
    
    elif ai_choice == '2':
        # PPO V1
        if not HAS_PPO:
            print("\n需要安裝 stable_baselines3")
            return
        
        if not available_models['ppo_v1']:
            print("\n沒有可用的 PPO V1 模型")
            print("請先訓練模型: python snake_ai_ppo.py --mode train")
            return
        
        print("\n可用的 PPO V1 模型:")
        for i, model in enumerate(available_models['ppo_v1'], 1):
            print(f"  {i}. {model}")
        
        if len(available_models['ppo_v1']) == 1:
            model_idx = 0
        else:
            model_choice = input(f"\n請選擇模型 (1-{len(available_models['ppo_v1'])}): ").strip()
            model_idx = int(model_choice) - 1
        
        model_path = available_models['ppo_v1'][model_idx]
        board_size = get_user_choice_board_size()
        
        print(f"\n載入模型: {model_path}")
        print(f"棋盤大小: {board_size}x{board_size}")
        
        watch_ppo_play(version='v1', board_size=board_size, model_path=model_path)
    
    elif ai_choice == '3':
        # PPO V2
        if not HAS_PPO:
            print("\n需要安裝 stable_baselines3")
            return
        
        if not available_models['ppo_v2']:
            print("\n沒有可用的 PPO V2 模型")
            print("請先訓練模型: python snake_ai_ppo_v2.py --mode train")
            return
        
        print("\n可用的 PPO V2 模型:")
        for i, model in enumerate(available_models['ppo_v2'], 1):
            print(f"  {i}. {model}")
        
        if len(available_models['ppo_v2']) == 1:
            model_idx = 0
        else:
            model_choice = input(f"\n請選擇模型 (1-{len(available_models['ppo_v2'])}): ").strip()
            model_idx = int(model_choice) - 1
        
        model_path = available_models['ppo_v2'][model_idx]
        board_size = get_user_choice_board_size()
        
        print(f"\n載入模型: {model_path}")
        print(f"棋盤大小: {board_size}x{board_size}")
        
        watch_ppo_play(version='v2', board_size=board_size, model_path=model_path)
    
    else:
        print("無效選擇")


def run_with_ui_selector():
    """使用圖形化 UI 選擇模型並運行演示"""
    print("\n" + "="*70)
    print("🐍 貪吃蛇 AI 演示程序 - 圖形化選擇界面")
    print("="*70)
    print("\n正在啟動模型選擇器...")
    
    # 運行選擇器
    selector = ModelSelector(width=900, height=700)
    model_type, model_path, board_size = selector.run()
    
    # 檢查是否取消
    if model_type is None:
        print("\n已取消選擇")
        return
    
    print(f"\n選擇的配置:")
    print(f"  模型類型: {model_type}")
    print(f"  模型路徑: {model_path}")
    print(f"  棋盤大小: {board_size}x{board_size}")
    print("\n正在啟動遊戲...")
    
    # 根據模型類型運行相應的演示
    if model_type == 'qlearning':
        watch_ai_play(board_size=board_size, model_path=model_path)
    elif model_type == 'ppo_v1':
        watch_ppo_play(version='v1', board_size=board_size, model_path=model_path)
    elif model_type == 'ppo_v2':
        watch_ppo_play(version='v2', board_size=board_size, model_path=model_path)
    elif model_type == 'ppo_v3':
        watch_ppo_play(version='v3', board_size=board_size, model_path=model_path)
    
def main():
    """主函數"""
    
    print("\n" + "="*70)
    print("🐍 貪吃蛇 AI 演示程序")
    print("="*70)
    
    # 顯示可用模型
    available_models = list_available_models()
    has_qlearning = len(available_models['qlearning']) > 0
    has_ppo_v1 = len(available_models['ppo_v1']) > 0
    has_ppo_v2 = len(available_models['ppo_v2']) > 0
    has_ppo_v3 = len(available_models['ppo_v3']) > 0
    
    print("\n可用的 AI 模型:")
    if has_qlearning:
        print(f"  ✓ Q-learning: {len(available_models['qlearning'])} 個模型")
    else:
        print(f"  ✗ Q-learning: 無可用模型")
    
    if HAS_PPO:
        if has_ppo_v1:
            print(f"  ✓ PPO V1: {len(available_models['ppo_v1'])} 個模型")
        else:
            print(f"  ✗ PPO V1: 無可用模型")
        
        if has_ppo_v2:
            print(f"  ✓ PPO V2: {len(available_models['ppo_v2'])} 個模型")
        else:
            print(f"  ✗ PPO V2: 無可用模型")
        
        if has_ppo_v3:
            print(f"  ✓ PPO V3 (課程學習): {len(available_models['ppo_v3'])} 個模型")
        else:
            print(f"  ✗ PPO V3: 無可用模型")
    else:
        print(f"  ✗ PPO: 需要安裝 stable_baselines3")
    
    print("\n" + "="*70)
    print("請選擇演示模式:")
    print()
    print("━━━ 圖形化選擇 (推薦) 🎨 ━━━")
    print("  0. 使用圖形化界面選擇模型 ⭐ NEW!")
    print()
    print("━━━ Q-learning 方法 (傳統強化學習) ━━━")
    print("  1. 觀看 Q-learning AI 遊戲 (可選棋盤大小)")
    print("  2. Q-learning 訓練前後比較")
    print("  3. Q-learning AI 策略分析")
    print()
    
    if HAS_PPO:
        print("━━━ PPO 方法 (深度強化學習) 🌟 ━━━")
        print("  4. 觀看 PPO V1 AI 遊戲 (可選棋盤大小)")
        print("  5. 觀看 PPO V2 AI 遊戲 (增強版，推薦) ⭐")
        print("  6. Q-learning vs PPO V1 vs PPO V2 三方比較")
        print("  7. 選擇特定模型和棋盤大小 (進階)")
    else:
        print("━━━ PPO 方法 (需要安裝 stable_baselines3) ━━━")
        print("  安裝方法: pip install stable-baselines3 torch")
    
    print()
    print("="*70)
    
    try:
        choice = input("\n請輸入選擇 (0-7): ").strip()
        
        if choice == '0':
            run_with_ui_selector()
        elif choice == '1':
            watch_ai_play()
        elif choice == '2':
            compare_before_after_training()
        elif choice == '3':
            analyze_ai_strategy()
        elif choice == '4':
            if HAS_PPO:
                watch_ppo_play(version='v1')
            else:
                print("\n請先安裝 stable_baselines3:")
                print("  pip install stable-baselines3 torch")
        elif choice == '5':
            if HAS_PPO:
                watch_ppo_play(version='v2')
            else:
                print("\n請先安裝 stable_baselines3:")
                print("  pip install stable-baselines3 torch")
        elif choice == '6':
            if HAS_PPO:
                compare_all_methods()
            else:
                print("\n請先安裝 stable_baselines3:")
                print("  pip install stable-baselines3 torch")
        elif choice == '7':
            if HAS_PPO:
                advanced_mode()
            else:
                print("\n請先安裝 stable_baselines3:")
                print("  pip install stable-baselines3 torch")
        else:
            print("無效選擇")
            
    except KeyboardInterrupt:
        print("\n程序被用戶中斷")

if __name__ == "__main__":
    main()