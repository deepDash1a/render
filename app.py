# create fastapi app
import os

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile

from character_recognition import readPlate2
from plate_detection import plate_detection

def read_plate(img):
    plate = plate_detection(img)
    text = readPlate2(plate)
    return text


app = FastAPI()


@app.post("/read_plate")
def read_plate_endpoint(img: UploadFile = File(...)):
    try:
        contents = img.file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return {"error": "Failed to decode image"}

        text = read_plate(image)
        return {"text": text}

    except Exception as e:
        return {"error": f"Error processing image: {str(e)}"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000)) 
    uvicorn.run(app, host="0.0.0.0", port=port)
