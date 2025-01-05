## Basic setup real quick

## i use venv
python -m upscaler-code

## 
chmod +x build_pip_package.sh
./build_pip_package.sh

## Run server inside of dir 
python server.py

## Test calls

curl -F "file=@PICT0028.jpg" http://localhost:5001/upscale



