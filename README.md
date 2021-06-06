# project-web-torch
Flask web application for translating CT brain scans to probable MRIs
Uses pretrained torch models in static folder for real time image to image translation
Input images should be a brain CT scan and in "png" format

instructions
pip install -r requirements.txt
python app.py

To run with wasgi in dev
activate env
uswgi dev.ini
application will run on http://localhost:9090/
