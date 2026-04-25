@echo off
echo ========================================
echo   HR AI RESUME SCREENER - LAUNCHER
echo ========================================
echo.
echo Démarrage de l'API FastAPI...
start "API FastAPI" cmd /k "python app.py"
echo.
echo Attente 5 secondes pour que l'API démarre...
timeout /t 5 /nobreak > nul
echo.
echo Démarrage de l'interface Gradio...
start "Gradio Interface" cmd /k "python gradio_app.py"
echo.
echo ========================================
echo   TOUT EST LANCÉ !
echo ========================================
echo.
echo API: http://127.0.0.1:8000/docs
echo Gradio: http://127.0.0.1:7860
echo.
echo Fermez les deux fenêtres pour arrêter.
pause