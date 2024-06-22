import uvicorn
from fastapi import FastAPI, File, UploadFile
import os
import pickle as pkl
import tensorflow as tf
from numba import cuda
# importing jinja2
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from starlette.requests import Request
from fastapi import Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

device = cuda.get_current_device()
from Audio import Preprocessor

from predict import audio_extractor, video_breaker

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# from pydantic import BaseModel

origins = [
    "http://localhost:8000",
    "http://localhost:8001",
    "http://localhost:8002",
    "http://localhost:8003",
    "http://localhost:8004",
    "http://localhost:5500",
    "http://localhost:5500",
    "http://localhost:9000",
    "http://localhost:9001",
    "http://localhost:9002",
]

app = FastAPI(
    title="Emotion Detection API",
    description="This is a simple API for emotion detection",
    version="1.0.0",
    allow_credentials=True,
    allow_methods="*",
    allow_headers="*",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins
)

classes = {
    0: "anger",
    1: "bored",
    2: "excited",
    3: "fear",
    4: "happy",
    5: "relax",
    6: "sad",
    7: "worry"
}

global emotions, path
emotions = [0, 0, 0, 0, 0, 0, 0, 0]
path = 'src/yolov5/runs/detect/exp/labels'


def read_text_file(file_path: str) -> str:
    for file in os.listdir(file_path):
        # print(file)
        with open(f'{file_path}/{file}', "r") as f:
            emotions[int(f.read(1))] += 1
    return f"Done {file_path}"


def happy_or_frail(percentage: list) -> str:
    healthy = ["happy", "relax", "excited"]
    frail = ["sad", "worry", "bored", "fear", "anger"]
    if classes[percentage.index(max(percentage))] in healthy:
        return "Healthy"
    elif classes[percentage.index(max(percentage))] in frail:
        return "Frail"


# fastapi app to get audio file and return valence and arousal
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    file_path = f"data/scraped-videos/{file.filename}"
    print(file_path)
    audio_extractor(file_path)
    video_breaker(file_path)

    preprocessor = Preprocessor()
    features = preprocessor.get_features('test/tiktok/test.mp3')
    features = features.reshape(1, -1)
    model = tf.keras.models.load_model('models/ANN.h5')
    onehot_encoder = pkl.load(open('data/onehot_encoder.pkl', 'rb'))
    prediction = model.predict(features)
    device.reset()

    """Yolov5"""

    # os.system(
    #     "python3 'src/yolov5/detect.py' --weights 'src/weights/best.pt' --source 'results/frames/' --data src/config/edm8.yaml --save-txt")
    read_text_file(path)
    # print(emotions)
    percentage = [i / sum(emotions) for i in emotions]
    # rprint(percentage)
    Audio = onehot_encoder.inverse_transform(prediction)[0][0]
    percentage_emotions_audio = prediction[0]
    Video = classes[emotions.index(max(emotions))]
    percentage_emotions_video = percentage
    final = {
        'Audio': Audio,
        'Emotions_Audio': percentage_emotions_audio,
        'Video': Video,
        'Emotion_Video': percentage_emotions_video,
        'Final': happy_or_frail(percentage)
    }
    # print(final)
    # return JSONResponse(content=jsonable_encoder(final))
    return f"{final}"


templates = templates = Jinja2Templates(directory="src/templates")

def get_emotion(file_name)-> str:
    file_path = f"data/scraped-videos/{file_name}"
    print(file_path)
    audio_extractor(file_path)
    video_breaker(file_path)

    preprocessor = Preprocessor()
    features = preprocessor.get_features('test/tiktok/test.mp3')
    features = features.reshape(1, -1)
    model = tf.keras.models.load_model('models/ANN.h5')
    onehot_encoder = pkl.load(open('data/onehot_encoder.pkl', 'rb'))
    prediction = model.predict(features)
    device.reset()

    """Yolov5"""

    # os.system(
    #     "python3 'src/yolov5/detect.py' --weights 'src/weights/best.pt' --source 'results/frames/' --data src/config/edm8.yaml --save-txt")
    read_text_file(path)
    # print(emotions)
    percentage = [i / sum(emotions) for i in emotions]
    # rprint(percentage)
    Audio = onehot_encoder.inverse_transform(prediction)[0][0]
    percentage_emotions_audio = prediction[0]
    Video = classes[emotions.index(max(emotions))]
    percentage_emotions_video = percentage
    final = {
        'Audio': Audio,
        'Emotions_Audio': percentage_emotions_audio,
        'Video': Video,
        'Emotion_Video': percentage_emotions_video,
        'Final': happy_or_frail(percentage)
    }
    # print(final)
    # return JSONResponse(content=jsonable_encoder(final))
    return f"{final}"

@app.post("/filename")
async def get_file_name(request : Request):
    # get file from form
    file = await request.form()
    # call predict here
    prediction = get_emotion(file['video'])
    print(prediction)
    return prediction


@app.get("/")
async def read_root(request: Request):
    # get file from form

    return templates.TemplateResponse("home.html", {"request": request})


if __name__ == "__main__":
    uvicorn.run(debug=True, app=app)
