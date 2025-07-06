from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse  # Add this line
import shutil
import requests

from gop_scores import GOP


app = FastAPI()

print('Loading model...')
gop = GOP()
print('model loaded...')

@app.post("/upload-audio")
async def upload_audio(audio: UploadFile = File(...), transcript: str = Form(...)):
    with open("audio.wav", "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)
    
    scores = GOP().forward(
        'audio.wav',
        transcript
    )

    return scores

@app.get("/audio")
async def get_audio():
    return FileResponse("audio.wav")

