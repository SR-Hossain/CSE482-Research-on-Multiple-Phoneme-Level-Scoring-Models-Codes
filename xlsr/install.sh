#!/bin/bash

python3 -m venv venv-wav2vec2
source venv-wav2vec2/bin/activate
pip install -r requirements.txt

if [ ! -d "model" ]; then
    python3 -c "
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-xlsr-53-espeak-cv-ft')
model = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-xlsr-53-espeak-cv-ft')
processor.save_pretrained('model')
model.save_pretrained('model')
"
    echo "Model downloaded"
else
    echo "Model already exists"
fi

echo "Installation completed"
