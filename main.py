import os
from uuid import uuid4
import ocrmypdf
import subprocess
from passporteye import read_mrz
import cv2, face_recognition
from mtcnn import MTCNN
from liveness import classify
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = './'
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def prep(doc_path, filename, tmp_path):
    if not os.path.isdir(tmp_path):
        os.makedirs(tmp_path)
    out_filepath = os.path.join(tmp_path, f'{filename}.png')
    prog = ['convert']
    opts = ['-deskew', '95%', '-density', '300']
    inpt = [doc_path]
    out = [out_filepath]
    subprocess.run(prog+opts+inpt+out)
    return out_filepath


def get_face_blob(filepath, out_name, tmp_dir, save=True, mirror=False):
    tmp_dir = os.path.dirname(filepath)
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    detector = MTCNN()
    face_det = None
    face_dets = sorted(detector.detect_faces(img), key=lambda t: t['confidence'])
    if face_dets:
        face_det = face_dets[0]
    if not face_det:
        raise RuntimeError(f'Face not found in {filepath}')
    x, y, w, h = face_det['box']
    face = img[y:y+h, x:x+w]
    if mirror:
        face = cv2.flip(face, 1)
    if save:
        out_filepath = os.path.join(tmp_dir, f'{out_name}.png')
        cv2.imwrite(out_filepath, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
    return face


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        front = request.files['front']
        back = request.files['back']
        true_selfie = request.files['true_selfie']
        fake_selfie = request.files['fake_selfie']
        front_path = os.path.join(app.config['UPLOAD_FOLDER'], front.filename)
        back_path = os.path.join(app.config['UPLOAD_FOLDER'], back.filename)
        true_selfie_path = os.path.join(app.config['UPLOAD_FOLDER'], true_selfie.filename)
        fake_selfie_path = os.path.join(app.config['UPLOAD_FOLDER'], fake_selfie.filename)
        front.save(front_path)
        back.save(back_path)
        true_selfie.save(true_selfie_path)
        fake_selfie.save(fake_selfie_path)
        out_dir = './tmp'
        dirname = uuid4().__str__()
        tmp_path = os.path.abspath(os.path.join(out_dir, dirname))
        back = prep(back_path, 'back', tmp_path)
        front = prep(front_path, 'front', tmp_path)
        true_selfie_path = prep(true_selfie_path, 'true_selfie', tmp_path)
        false_selfie_path = prep(fake_selfie_path, 'false_selfie', tmp_path)
        mrz = read_mrz(back).to_dict()
        doc_face = get_face_blob(front, 'front_face', tmp_path)
        true_selfie = get_face_blob(true_selfie_path, 'true_selfie_face', tmp_path, mirror=True)
        true_selfie = cv2.cvtColor(true_selfie, cv2.COLOR_RGB2BGR)
        false_selfie = get_face_blob(false_selfie_path, 'false_selfie_face', tmp_path, mirror=True)
        false_selfie = cv2.cvtColor(false_selfie, cv2.COLOR_RGB2BGR)
        doc_encoding = face_recognition.face_encodings(doc_face, num_jitters=10, model='large')
        selfie_encoding = face_recognition.face_encodings(true_selfie, num_jitters=10, model='large')[0]
        dist = face_recognition.face_distance(doc_encoding, selfie_encoding)[0]
        pred1 = classify(true_selfie)
        pred2 = classify(false_selfie)

        return {
            'mrz_data': mrz,
            'similarity': (1 - dist)*100,
            'prediction_for_original': pred1,
            'prediction_for_fake': pred2
        }



if __name__ == "__main__":
    app.run('0.0.0.0', port=5000)