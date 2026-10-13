import torch
import transformers.pytorch_utils as pu

def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
    mask = torch.ones(n_heads, head_size)
    heads = set(heads) - already_pruned_heads
    for head in heads:
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    return heads, torch.arange(len(mask))[mask].long()

pu.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices

from surya.ocr import run_ocr
from surya.model.detection.model import load_model as load_det_model, load_processor as load_det_processor
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor
from PIL import Image

print("Loading models onto CUDA/GPU...")
det_processor, det_model = load_det_processor(), load_det_model()
rec_model, rec_processor = load_rec_model(), load_rec_processor()
print("Models loaded successfully onto device!")

img = Image.open('test_sample_page.png')
print("Running OCR on test_sample_page.png...")
predictions = run_ocr([img], [['en']], det_model, det_processor, rec_model, rec_processor)
print("OCR Completed! Found lines:", len(predictions[0].text_lines))
for line in predictions[0].text_lines[:5]:
    print(" - Line:", line.text)

