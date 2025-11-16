@echo off
echo Auto-pushing to GitHub...
echo.

git add .
git commit -m "Auto-push: %date% %time%"
git push origin main --force

echo.
echo Push completed!
pause
