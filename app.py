from flask import Flask, request, jsonify, send_file, send_from_directory
import os
import random
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from omegaconf import OmegaConf
from PIL import Image
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d_echo import EchoUNet3DConditionModel
from src.models.whisper.audio2feature import load_audio_model
from src.pipelines.pipeline_echo_mimic import Audio2VideoPipeline
from src.utils.util import save_videos_grid, crop_and_pad
from src.models.face_locator import FaceLocator
from moviepy.editor import VideoFileClip, AudioFileClip
from facenet_pytorch import MTCNN
from webgui import process_video
from flask_cors import CORS, cross_origin

app = Flask(__name__)

# 设置上传文件的目录
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 设置生成视频的目录
OUTPUT_FOLDER = 'output/tmp'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/upload_image', methods=['POST'])
@cross_origin()
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file part'}), 400

    image_file = request.files['image']

    if image_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # 生成唯一的文件名
    image_filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{image_file.filename}"
    image_path = os.path.join(UPLOAD_FOLDER, image_filename)

    image_file.save(image_path)

    return jsonify({
        'image_path': image_path
    }), 200

@app.route('/upload_audio', methods=['POST'])
@cross_origin()
def upload_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file part'}), 400

    audio_file = request.files['audio']

    if audio_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # 生成唯一的文件名
    audio_filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{audio_file.filename}"
    audio_path = os.path.join(UPLOAD_FOLDER, audio_filename)

    audio_file.save(audio_path)

    return jsonify({
        'audio_path': audio_path
    }), 200

@app.route('/generate_video', methods=['POST'])
@cross_origin()
def generate_video_http():
    data = request.form
    uploaded_img = data.get('uploaded_img')
    uploaded_audio = data.get('uploaded_audio')
    width = int(data.get('width', 512))
    height = int(data.get('height', 512))
    length = int(data.get('length', 1200))
    seed = int(data.get('seed', 420))
    facemask_dilation_ratio = float(data.get('facemask_dilation_ratio', 0.1))
    facecrop_dilation_ratio = float(data.get('facecrop_dilation_ratio', 0.5))
    context_frames = int(data.get('context_frames', 12))
    context_overlap = int(data.get('context_overlap', 3))
    cfg = float(data.get('cfg', 2.5))
    steps = int(data.get('steps', 30))
    sample_rate = int(data.get('sample_rate', 16000))
    fps = int(data.get('fps', 24))
    device = data.get('device', 'cuda')
    # 检查上传的音频或图片是否为空
    if not uploaded_img or not uploaded_audio:
        default_video_url = 'default_video.mp4'  
        return jsonify({
            'video_url': default_video_url
        }), 200
    final_output_path = process_video(
        uploaded_img, uploaded_audio, width, height, length, seed, facemask_dilation_ratio, facecrop_dilation_ratio, context_frames, context_overlap, cfg, steps, sample_rate, fps, device
    )
    
    return jsonify({
        'video_url': final_output_path
    }), 200
    # return jsonify({
    #         'video_url': 'output_video_with_audio_20241205145917.mp4'
    #     }), 200

@app.route('/video/<filename>')
@cross_origin()
def serve_video(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)