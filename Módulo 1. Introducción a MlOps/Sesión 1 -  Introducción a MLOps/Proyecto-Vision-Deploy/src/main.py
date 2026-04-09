import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import io
import cv2
import numpy as np
import cvlib as cv
from enum import Enum
from cvlib.object_detection import draw_bbox
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse

from magnum import Magnum

app = FastAPI(title='Deploying un Modelo ML con FastAPI')

class Model(str, Enum):
    yolov3tiny = "yolov3-tiny"
    yolov3 = "yolov3"


@app.get("/")
def home():
    return "Congratulations! Your API is working as expected."


@app.post("/predict")
def prediction(model: Model, file: UploadFile = File(...)):

    # 1. VALIDATE INPUT FILE
    filename = file.filename
    fileExtension = filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not fileExtension:
        raise HTTPException(status_code=415, detail="Unsupported file provided.")

    # 2. TRANSFORM RAW IMAGE INTO CV2 image
    image_stream = io.BytesIO(file.file.read())
    image_stream.seek(0)
    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # 3. RUN OBJECT DETECTION MODEL
    bbox, label, conf = cv.detect_common_objects(image, confidence=0.1, model=model)
    output_image = draw_bbox(image, bbox, label, conf)

    # 4. ENCODE AND RETURN IMAGE
    _, img_encoded = cv2.imencode('.jpg', output_image)
    return StreamingResponse(
        io.BytesIO(img_encoded.tobytes()),
        media_type="image/jpeg"
    )

handler = Magnum(app)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
