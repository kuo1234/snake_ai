@echo off
echo ========================================
echo 🎯 貪吃蛇AI簡報生成工具
echo ========================================
echo.

echo 📊 正在生成PowerPoint簡報...
C:/Python312/python.exe presentation_generator.py

echo.
echo 📖 正在顯示簡報內容摘要...
C:/Python312/python.exe presentation_viewer.py

echo.
echo ✅ 完成！
echo 📂 簡報文件已生成：snake_ai_presentation.pptx
echo 💡 請使用PowerPoint或相容軟體開啟檔案
echo.
pause