@echo off
SET BACKEND_HOST=172.20.10.4
SET BACKEND_PORT=8080
SET MQTT_HOST=172.20.10.4
SET MQTT_PORT=1883
SET MQTT_WS_PORT=9091
SET ESP32_IP=172.20.10.2
SET ESP32_PORT=81
SET CAM_STREAM=http://172.20.10.2:81/stream
SET CNN_MODEL=best_model.h5
SET XGB_MODEL=modele_ded_xgboost.pkl
SET TF_CPP_MIN_LOG_LEVEL=2
SET TF_ENABLE_ONEDNN_OPTS=0
cd /d "c:\Users\PC\Desktop\ded-monitor\web\backend"
"c:\Users\PC\Desktop\ded-monitor\web\backend\venv\Scripts\uvicorn.exe" main:app --host 0.0.0.0 --port 8080
