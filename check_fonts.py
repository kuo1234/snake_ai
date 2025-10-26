import pygame

pygame.init()

# 獲取所有可用字體
all_fonts = pygame.font.get_fonts()

# 篩選中文字體
chinese_fonts = [f for f in all_fonts if any(keyword in f.lower() for keyword in 
    ['yahei', 'sim', 'kai', 'hei', 'ming', 'song', 'fang', 'gothic'])]

print("系統可用的中文字體:")
for font in chinese_fonts:
    print(f"  - {font}")

if not chinese_fonts:
    print("  (未找到常見中文字體)")

print(f"\n總共找到 {len(chinese_fonts)} 個中文字體")
print(f"系統總字體數: {len(all_fonts)}")

# 測試加載
print("\n測試字體渲染:")
test_fonts = ['microsoftyahei', 'microsoftyaheiui', 'simsun', 'simhei']
for font_name in test_fonts:
    try:
        font = pygame.font.SysFont(font_name, 24)
        test_surface = font.render('測試中文', True, (255, 255, 255))
        if test_surface.get_width() > 0:
            print(f"  ✓ {font_name}: 成功")
        else:
            print(f"  ✗ {font_name}: 失敗（寬度為0）")
    except Exception as e:
        print(f"  ✗ {font_name}: 失敗 ({e})")

pygame.quit()
