# _*_coding : UTF-8_*_
import os
import glob
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

from tools.render import Renderer
import tools.log as log
from lib.BFM.model_basis.BFM2009.face_generate import Face3D
from network.MVRnet import MVRNet
from network.Bisenet import BiSeNet
from config.config_test import get_parser
from dataset.pixel_face.dataset_preprocess import get_face_mask

def visualize_result(ret, f_name, result_path):
    os.makedirs(result_path, exist_ok=True)
    
    # Renderers in code use 224 usually, we try 512 for detail
    renderer = Renderer(img_size=512)
    
    face_shape = ret['face_shape_ft'] # [1, 35709, 3]
    face_texture = ret['face_texture'] # [1, 35709, 3]
    
    face_model = Face3D()
    cells = face_model.facemodel.cell # [70766, 3]
    if torch.is_tensor(cells):
        cells_torch = cells.unsqueeze(0).cuda()
    else:
        cells_torch = torch.from_numpy(cells).unsqueeze(0).cuda()
    
    # Render from model
    with torch.no_grad():
        render_img, render_mask, _ = renderer(face_shape, face_texture, cells_torch)
    
    # Convert to image
    img_np = render_img[0].cpu().numpy()
    img_np = np.clip(img_np, 0, 1)
    
    # Save image
    plt.figure(figsize=(10, 10))
    plt.imshow(img_np)
    plt.title(f"Rendered Result: {f_name}")
    plt.axis('off')
    save_img_path = os.path.join(result_path, f"{f_name}_rendered.png")
    plt.savefig(save_img_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Detailed visualization saved to: {save_img_path}")

def run_detailed_test():
    cfg = get_parser()
    gpu = 0
    torch.cuda.set_device(gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # Setup Paths
    result_path_vis = 'samples_results/vis/'
    os.makedirs(result_path_vis, exist_ok=True)
    
    # Face Mask Model
    mask_model = BiSeNet(10)
    mask_model.cuda()
    mask_model.load_state_dict(torch.load('pretrain/face_mask.pth'))
    mask_model.eval()
    
    # MVR Model
    model = MVRNet(cfg).cuda()
    pretrain_file = 'pretrain/000000279.pth'
    checkpoint = torch.load(pretrain_file, map_location=lambda storage, loc: storage.cuda(gpu))
    
    # Handle DataParallel/DistributedDataParallel prefix 'module.'
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)
    model.eval()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    files_list = glob.glob('samples/1/*.png')
    for f_i, f_path in enumerate(files_list[:2]): # Just top 2 for speed
        f_name = f_path.split('/')[-1][:-5]
        print(f"Processing {f_name}...")
        
        fp_0 = f'samples/0/{f_name}0.png'
        fp_1 = f'samples/1/{f_name}1.png'
        fp_2 = f'samples/2/{f_name}2.png'
        
        imgs = [np.array(Image.open(p)) for p in [fp_0, fp_1, fp_2]]
        norms = [transform(img.copy()) for img in imgs]
        
        inputs = []
        for norm in norms:
            mask, _ = get_face_mask(norm, mask_model)
            mask_t = torch.from_numpy(mask[np.newaxis, np.newaxis, ...]).type(torch.float32)
            inp = torch.cat((norm.unsqueeze(0), mask_t), dim=1).cuda()
            inputs.append(inp)
            
        with torch.no_grad():
            ret = model(inputs[0], inputs[1], inputs[2])
            
        visualize_result(ret, f_name, result_path_vis)

if __name__ == '__main__':
    run_detailed_test()
