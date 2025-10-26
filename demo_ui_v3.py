"""
貪吃蛇 AI 可視化演示 (V3 Dynamic Rewards 版本)
觀看訓練好的 PPO V3 課程學習模型如何遊戲
支援 Stage 1-4 動態獎勵系統
"""

import sys
import time
import os
import glob
import pygame
import numpy as np

# 導入 PPO 和 V3 環境
try:
    from stable_baselines3 import PPO
    from envs.gym_snake_env_v3 import GymSnakeEnvV3
    HAS_PPO = True
except ImportError:
    HAS_PPO = False
    print("錯誤: 需要 stable_baselines3 和 gym_snake_env_v3")
    sys.exit(1)

# UI 顏色定義
COLOR_BG = (20, 20, 30)
COLOR_TITLE = (255, 255, 255)
COLOR_TEXT = (200, 200, 200)
COLOR_BUTTON = (50, 100, 150)
COLOR_BUTTON_HOVER = (70, 130, 180)
COLOR_BUTTON_SELECTED = (100, 180, 100)
COLOR_BORDER = (100, 100, 120)
COLOR_INFO = (255, 215, 0)


def get_chinese_font(size):
    """獲取支持中文的字體"""
    chinese_fonts = [
        'microsoftyahei',      # 微軟雅黑
        'microsoftyaheiui',    # 微軟雅黑 UI
        'simsun',              # 宋體
        'simhei',              # 黑體
        'kaiti',               # 楷體
        'arial',               # Arial Unicode MS
    ]
    
    for font_name in chinese_fonts:
        try:
            font = pygame.font.SysFont(font_name, size)
            test_surface = font.render('測試', True, (255, 255, 255))
            if test_surface.get_width() > 0:
                return font
        except:
            continue
    
    print("警告: 無法加載中文字體，可能會顯示亂碼")
    return pygame.font.Font(None, size)


def list_v3_models():
    """掃描並列出所有 V3 課程學習模型"""
    models = {
        'stage1': [],
        'stage2': [],
        'stage3': [],
        'stage4': []
    }
    
    ppo_v3_base = "models/ppo_snake_v3_curriculum"
    if not os.path.exists(ppo_v3_base):
        print(f"警告: 找不到 V3 模型目錄: {ppo_v3_base}")
        return models
    
    stages = [
        ("Stage1_Novice", "stage1", 6),
        ("Stage2_Intermediate", "stage2", 8),
        ("Stage3_Advanced", "stage3", 10),
        ("Stage4_Master", "stage4", 12)
    ]
    
    for stage_name, stage_key, board_size in stages:
        stage_dir = os.path.join(ppo_v3_base, stage_name)
        if os.path.exists(stage_dir):
            # 最佳模型（畢業模型）
            best = os.path.join(stage_dir, "best_model", "best_model.zip")
            if os.path.exists(best):
                models[stage_key].append({
                    'path': best,
                    'name': f"{stage_name} - 最佳模型",
                    'board_size': board_size,
                    'stage_num': int(stage_key[-1])
                })
            
            # 最終模型
            final = os.path.join(stage_dir, "model.zip")
            if os.path.exists(final):
                models[stage_key].append({
                    'path': final,
                    'name': f"{stage_name} - 最終模型",
                    'board_size': board_size,
                    'stage_num': int(stage_key[-1])
                })
    
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
        if self.is_selected:
            color = COLOR_BUTTON_SELECTED
        elif self.is_hovered:
            color = COLOR_BUTTON_HOVER
        else:
            color = COLOR_BUTTON
        
        pygame.draw.rect(screen, color, self.rect, border_radius=8)
        pygame.draw.rect(screen, COLOR_BORDER, self.rect, 2, border_radius=8)
        
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


class V3ModelSelector:
    """V3 模型選擇器 UI"""
    def __init__(self, width=900, height=700):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("PPO V3 課程學習模型選擇器")
        
        self.font_large = get_chinese_font(48)
        self.font_medium = get_chinese_font(32)
        self.font_small = get_chinese_font(22)
        self.font_tiny = get_chinese_font(18)
        
        self.selected_stage = None
        self.selected_model = None
        self.models = list_v3_models()
        self.stage_buttons = []
        self.model_buttons = []
        self.start_button = None
        
        self.setup_stage_selection()
    
    def setup_stage_selection(self):
        """設置階段選擇按鈕"""
        button_width = 180
        button_height = 70
        start_y = 180
        spacing = 90
        start_x = (self.width - (button_width * 2 + 40)) // 2
        
        stages = [
            ('stage1', 'Stage 1: 6x6', '新手村'),
            ('stage2', 'Stage 2: 8x8', '進階班'),
            ('stage3', 'Stage 3: 10x10', '挑戰班'),
            ('stage4', 'Stage 4: 12x12', '大師班')
        ]
        
        self.stage_buttons = []
        for i, (stage_key, stage_name, desc) in enumerate(stages):
            row = i // 2
            col = i % 2
            x = start_x + col * (button_width + 40)
            y = start_y + row * spacing
            
            count = len(self.models[stage_key])
            text = f"{stage_name}\n({count}個模型)"
            btn = UIButton(x, y, button_width, button_height, text, stage_key)
            self.stage_buttons.append(btn)
    
    def setup_model_selection(self):
        """設置模型選擇按鈕"""
        if not self.selected_stage:
            return
        
        button_width = 500
        button_height = 60
        start_y = 200
        spacing = 70
        center_x = self.width // 2 - button_width // 2
        
        self.model_buttons = []
        models = self.models[self.selected_stage]
        
        for i, model_info in enumerate(models):
            y = start_y + i * spacing
            btn = UIButton(center_x, y, button_width, button_height, 
                          model_info['name'], model_info)
            self.model_buttons.append(btn)
        
        # 開始按鈕
        if models:
            start_y_pos = start_y + len(models) * spacing + 30
            self.start_button = UIButton(center_x + 150, start_y_pos, 200, 60, 
                                        "開始演示", "start")
    
    def draw_stage_selection(self):
        """繪製階段選擇頁面"""
        self.screen.fill(COLOR_BG)
        
        # 標題
        title = self.font_large.render("PPO V3 課程學習", True, COLOR_TITLE)
        title_rect = title.get_rect(center=(self.width // 2, 60))
        self.screen.blit(title, title_rect)
        
        subtitle = self.font_small.render("選擇訓練階段", True, COLOR_TEXT)
        subtitle_rect = subtitle.get_rect(center=(self.width // 2, 120))
        self.screen.blit(subtitle, subtitle_rect)
        
        # 繪製階段按鈕
        for btn in self.stage_buttons:
            btn.draw(self.screen, self.font_small)
        
        # 說明文字
        info_y = 450
        info_texts = [
            "動態獎勵系統:",
            "Stage 1: 保守策略（邊緣移動，耐心）",
            "Stage 2: 積極策略（主動追食，飢餓懲罰）",
            "Stage 3: 進階策略（空間管理，強化中心）",
            "Stage 4: 大師策略（平衡策略，最高難度）"
        ]
        
        for i, text in enumerate(info_texts):
            color = COLOR_INFO if i == 0 else COLOR_TEXT
            info = self.font_tiny.render(text, True, color)
            info_rect = info.get_rect(center=(self.width // 2, info_y + i * 30))
            self.screen.blit(info, info_rect)
        
        pygame.display.flip()
    
    def draw_model_selection(self):
        """繪製模型選擇頁面"""
        self.screen.fill(COLOR_BG)
        
        # 標題
        stage_names = {
            'stage1': 'Stage 1 (6x6) - 新手村',
            'stage2': 'Stage 2 (8x8) - 進階班',
            'stage3': 'Stage 3 (10x10) - 挑戰班',
            'stage4': 'Stage 4 (12x12) - 大師班'
        }
        
        title = self.font_large.render(stage_names[self.selected_stage], True, COLOR_TITLE)
        title_rect = title.get_rect(center=(self.width // 2, 60))
        self.screen.blit(title, title_rect)
        
        subtitle = self.font_small.render("選擇模型", True, COLOR_TEXT)
        subtitle_rect = subtitle.get_rect(center=(self.width // 2, 130))
        self.screen.blit(subtitle, subtitle_rect)
        
        # 繪製模型按鈕
        for btn in self.model_buttons:
            btn.draw(self.screen, self.font_small)
        
        # 繪製開始按鈕
        if self.start_button and self.selected_model:
            self.start_button.draw(self.screen, self.font_medium)
        
        # 返回按鈕
        back_text = self.font_tiny.render("← 返回階段選擇", True, COLOR_TEXT)
        self.screen.blit(back_text, (20, 20))
        
        pygame.display.flip()
    
    def run(self):
        """運行選擇器"""
        clock = pygame.time.Clock()
        running = True
        page = 'stage'  # 'stage' or 'model'
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        if page == 'model':
                            page = 'stage'
                            self.selected_stage = None
                            self.selected_model = None
                        else:
                            return None
                
                if page == 'stage':
                    # 處理階段選擇
                    for btn in self.stage_buttons:
                        if btn.handle_event(event):
                            self.selected_stage = btn.value
                            page = 'model'
                            self.setup_model_selection()
                            for b in self.stage_buttons:
                                b.is_selected = (b == btn)
                
                elif page == 'model':
                    # 處理模型選擇
                    for btn in self.model_buttons:
                        if btn.handle_event(event):
                            self.selected_model = btn.value
                            for b in self.model_buttons:
                                b.is_selected = (b == btn)
                    
                    # 處理開始按鈕
                    if self.start_button and self.selected_model:
                        if self.start_button.handle_event(event):
                            pygame.quit()
                            return self.selected_model
                    
                    # 處理返回
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        if event.pos[1] < 50:
                            page = 'stage'
                            self.selected_stage = None
                            self.selected_model = None
            
            # 繪製當前頁面
            if page == 'stage':
                self.draw_stage_selection()
            else:
                self.draw_model_selection()
            
            clock.tick(30)
        
        pygame.quit()
        return None


def run_demo_v3(model_info):
    """運行 V3 模型演示"""
    if not model_info:
        print("未選擇模型")
        return
    
    model_path = model_info['path']
    board_size = model_info['board_size']
    stage_num = model_info['stage_num']
    
    print(f"\n{'='*70}")
    print(f"載入模型: {model_info['name']}")
    print(f"模型路徑: {model_path}")
    print(f"棋盤大小: {board_size}x{board_size}")
    print(f"訓練階段: Stage {stage_num}")
    print(f"{'='*70}\n")
    
    # 載入模型
    try:
        model = PPO.load(model_path)
        print("✓ 模型載入成功")
    except Exception as e:
        print(f"✗ 模型載入失敗: {e}")
        return
    
    # 創建環境（使用對應的 stage 參數）
    curriculum_stage = "conservative" if stage_num == 1 else "aggressive"
    env = GymSnakeEnvV3(
        board_size=board_size, 
        render_mode="human",
        curriculum_stage=curriculum_stage,
        stage=stage_num
    )
    
    print(f"✓ 環境創建成功 (Stage {stage_num})")
    print(f"\n開始演示... (按 ESC 退出)")
    print(f"{'='*70}\n")
    
    episode = 0
    try:
        while True:
            episode += 1
            print(f"回合 {episode}")
            
            obs, info = env.reset()
            done = False
            step_count = 0
            total_reward = 0
            
            while not done:
                # 預測動作
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                total_reward += reward
                step_count += 1
                
                # 渲染
                env.render()
                time.sleep(0.05)
                
                # 檢查退出
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise KeyboardInterrupt
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            raise KeyboardInterrupt
            
            score = info.get('score', 0)
            print(f"  分數: {score}, 步數: {step_count}, 總獎勵: {total_reward:.2f}")
            
            # 短暫暫停
            time.sleep(1.0)
    
    except KeyboardInterrupt:
        print(f"\n演示結束")
    finally:
        env.close()


def main():
    """主函數"""
    if not HAS_PPO:
        print("錯誤: 需要安裝 stable_baselines3")
        print("安裝命令: pip install stable-baselines3")
        return
    
    # 顯示選擇器
    selector = V3ModelSelector()
    model_info = selector.run()
    
    if model_info:
        # 運行演示
        run_demo_v3(model_info)
    else:
        print("未選擇模型，退出")


if __name__ == "__main__":
    main()
