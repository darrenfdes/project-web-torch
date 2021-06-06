# project-web-torch
Flask web application for translating CT brain scans to probable MRIs
Uses pretrained torch models in static folder for real time image to image translation
Input images should be a brain CT scan and in "png" format

instructions
create virtual env int directory (optional) and activate
1.pip install -r requirements.txt
2.python app.py

To run with wasgi in dev
activate env
1.uswgi dev.ini
2.application will run on http://localhost:9090/
