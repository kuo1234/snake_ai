"""
測試 UI 中文字體顯示
"""
import pygame
import sys
sys.path.insert(0, '.')
from demo_ai import get_chinese_font, COLOR_BG, COLOR_TITLE, COLOR_TEXT

pygame.init()

# 創建窗口
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("中文字體測試")

# 獲取字體
font_large = get_chinese_font(48)
font_medium = get_chinese_font(36)
font_small = get_chinese_font(24)

# 測試文字
texts = [
    (font_large, "選擇 AI 模型類型", 100),
    (font_medium, "Q-Learning (3個)", 180),
    (font_medium, "PPO V1 (4個)", 240),
    (font_medium, "PPO V2 (6個)", 300),
    (font_small, "使用滑鼠點擊選擇", 400),
    (font_small, "按 ESC 返回上一步", 450),
]

# 主循環
clock = pygame.time.Clock()
running = True

print("顯示中文字體測試窗口...")
print("按 ESC 或關閉窗口退出")

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
    
    # 清空屏幕
    screen.fill(COLOR_BG)
    
    # 繪製文字
    for font, text, y in texts:
        text_surface = font.render(text, True, COLOR_TITLE)
        text_rect = text_surface.get_rect(center=(400, y))
        screen.blit(text_surface, text_rect)
    
    # 更新顯示
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
print("測試完成！")
