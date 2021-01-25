import cv2
import torch
from skvideo.io import FFmpegWriter, vreader
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor

from common.facedetector import FaceDetector
from train import MaskDetector

from flask import Flask, request, jsonify, Response
import threading
from queue import Queue

app = Flask(__name__, static_folder="static")

global_queue = Queue()

@app.route('/detect_mask', methods=['POST'])
def detect():
    global global_queue
    args = request.json
    # demo_id = args['demo_id']
    path = args['path']
    global_queue.put(path)
    result_dict = {"status": 1}
    return jsonify(result_dict)



def tagVideo(videopath, modelpath='models/face_mask.ckpt', nameOutput='None.mp4'):
    model = MaskDetector()
    model.load_state_dict(torch.load(modelpath)['state_dict'], strict=False)
    
    device = torch.device("cpu")
    model = model.to(device)
    model.eval()
    
    faceDetector = FaceDetector(
        prototype='models/deploy.prototxt.txt',
        model='models/res10_300x300_ssd_iter_140000.caffemodel',
    )
    
    transformations = Compose([
        ToPILImage(),
        Resize((100, 100)),
        ToTensor(),
    ])

    outputPath = 'output/'+nameOutput

    writer = FFmpegWriter(outputPath)

    cur_frame = 0
    bboxs = []
    frames = []

    for frame in vreader(str(videopath)):
        cur_frame += 1
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = faceDetector.detect(frame)
        for face in faces:
            xStart, yStart, width, height = face
            xStart, yStart = max(xStart, 0), max(yStart, 0)
            faceImg = frame[yStart:yStart+height, xStart:xStart+width]
            try:
                output = model(transformations(faceImg).unsqueeze(0).to(device))
            except: continue
            _, predicted = torch.max(output.data, 1)

            if int(predicted) == 0:
                frames.append(cur_frame)
                bboxs.append([xStart, yStart, xStart + width, yStart + height])
                writer.writeFrame(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.close()
    cv2.destroyAllWindows()
    return {'outputPath': outputPath, 'frames': frames, 'bboxs': bboxs}


def progress():
    global global_queue
    while True:
        if not global_queue.empty():
            print('dang xu ly')
            path = global_queue.get()
            try:
                print(tagVideo(videopath=path))
            except Exception as e:
                print(e)
                pass


if __name__ == '__main__':
    progress_thread = threading.Thread(target=progress)
    progress_thread.start()
    app.run(debug=True)
    # tagVideo('data/khanh2.mp4')
