# project-web-torch
Flask web application for translating CT brain scans to probable MRIs
Uses pretrained torch models in static folder for real time image to image translation
Input images should be a brain CT scan and in "png" format

instructions
create virtual env int directory (optional) and activate \n
1.pip install -r requirements.txt \n
2.python app.py \n

To run with wasgi in dev \n
activate env \n
1.uswgi dev.ini \n
2.application will run on http://localhost:9090/
