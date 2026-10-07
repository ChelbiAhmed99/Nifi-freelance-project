we should modify the dtbc in nifi to update the path then start mysql server 

we should update oce_service.py to uncomment the path


python -m venv venv
venv\Scripts\activate
pip install flask opencv-python-headless pytesseract pillow

installer le modele https://github.com/UB-Mannheim/tesseract/wiki


python ocr_service.py
