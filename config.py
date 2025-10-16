# 游戏配置设置

# 默认游戏设置
BOARD_SIZE = 12          # 游戏板大小 (12x12)
CELL_SIZE = 40          # 每个格子的像素大小
UPDATE_INTERVAL = 0.15  # 游戏更新间隔（秒）
BORDER_SIZE = 20        # 边框大小

# 颜色设置 (RGB格式)
COLORS = {
    'background': (0, 0, 0),           # 黑色背景
    'border': (255, 255, 255),         # 白色边框
    'snake_head': (100, 100, 255),     # 蓝色蛇头
    'snake_body': (0, 255, 0),         # 绿色蛇身
    'snake_tail': (255, 100, 100),     # 红色蛇尾
    'food': (255, 0, 0),               # 红色食物
    'text': (255, 255, 255),           # 白色文字
    'button_normal': (100, 100, 100),  # 按钮正常颜色
    'button_hover': (255, 255, 255),   # 按钮悬停颜色
}

# 分数设置
FOOD_SCORE = 10  # 每个食物的分数

# 音效文件路径
SOUND_FILES = {
    'eat': 'sound/eat.wav',
    'game_over': 'sound/game_over.wav',
    'victory': 'sound/victory.wav',
}

# 字体设置
FONT_SIZE = 36