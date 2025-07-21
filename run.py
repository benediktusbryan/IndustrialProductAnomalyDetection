import argparse
from tqdm import tqdm
import os
import copy

import torch
import torch.nn.functional as F
import numpy as np
import cv2 as cv

from src import tools, Cfg, template as tl

device = 0
torch.cuda.set_device(device)

def argParse():

    parser = argparse.ArgumentParser(description='ANOMALYDETECTION')
    parser.add_argument('--mode', choices=['test', 'temp'], default='test')
    parser.add_argument('--method', default='ATMM')
    parser.add_argument('--ttype', choices=['ALL', 'PTS'], default='ALL')
    parser.add_argument('--tsize', type=int, default=0)
    parser.add_argument('--datapath', help='your own data path')
    parser.add_argument('--dataset', type=str, default='MVTec_AD')
    parser.add_argument('--categories', type=str, nargs='+', default=None)
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--save_map', action='store_true')
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--silence', action='store_true')

    args = parser.parse_args()
    return args

def test(cfg):
    rpath = os.path.join(cfg.rpath, cfg.dataset, cfg.ttype, cfg.category)
    
    with torch.no_grad():
        scores, gts, imgs, fg_masks = [], [], [], []
        for batch in cfg.testloader:
            img, gt = batch
            query = cfg.model(img.cuda())    
            score = cfg.model.impl(query, cfg.temp)          
            features = query.get('layer1')  # [B, C, H, W]

            # FEB Process for Spesific Category
            if hasattr(cfg, "feb") and cfg.feb is not None and cfg.category in ['bottle', 'capsule', 'metal_nut', 'toothbrush']:   
                print("feb")
                bs = img.size(0)
                for b in range(bs):
                    feature = features[b]  # # shape: [C, H, W]
                    fg_mask = cfg.feb(feature.unsqueeze(0))[0, 0].cpu().numpy()

                    # Resize foreground mask to score shape
                    fg_mask_resized = cv.resize(fg_mask, (score.shape[-1], score.shape[-2]))  # (W, H)
    
                    # preprocessing
                    cleaned = (fg_mask_resized > 0.3).astype(np.uint8)
                    fg_mask = cleaned.astype(np.float32)
                    fg_masks.append((fg_mask * 255).astype(np.uint8))  # Scale to [0, 255]                 
                    fg_mask_tensor = torch.from_numpy(fg_mask).to(score.device)

                    min_val = score[b, 0].min()
                    score[b, 0][~fg_mask_tensor.bool()] = min_val
            scores.append(score), gts.append(gt), imgs.append(cfg.testset.inv_trans(img))
            
        scores = torch.cat(scores, 0)
        ans = cfg.model.post_process(scores)
        ans = {k : v if 'img_AUC' in k else torch.squeeze(v, 1) for k, v in ans.items()}               
        gts = tools.binarize(torch.squeeze(torch.cat(gts, 0), 1))
        gls = torch.tensor(cfg.testset.labels) 
        imgs = torch.cat(imgs, 0)

        cfg.metrics.evaluate(cfg.category, gls, gts, ans)
        if cfg.save_map:
            tools.save_anomaly_map(ans['pix_AUC'], imgs, gts, rpath, cfg.testset.filenames, 'HETMM', cfg.testset.types)

            save_fg_mask(fg_masks, rpath, cfg.testset.filenames, cfg.testset.types)

def save_fg_mask(fg_masks, rpath, filenames, types):
    # Generate the file path for saving
    for i in range(len(fg_masks)):
        img_name = f"{types[i]}_{filenames[i]}_fg.png"                       # 'broken_large_000_fg.png'
        fg_save_path = os.path.join(rpath, img_name)
        # Save the foreground mask as a PNG file
        cv.imwrite(fg_save_path, fg_masks[i])

def temp(cfg):
    tpath = os.path.join(cfg.tpath, cfg.dataset, cfg.category)
    tname = f'{cfg.model.backbone.lower()}_ALL.pkl' if cfg.ttype == 'ALL' else f'{cfg.model.backbone.lower()}_{cfg.ttype}x{cfg.tsize}.pkl'
    os.makedirs(tpath, exist_ok=True)

    def get_ALL(cfg, tpath):
        try:
            tdict = cfg.model.load_template(os.path.join(tpath, f'{cfg.model.backbone.lower()}_ALL.pkl'))

        except:
            tdict = tl.gen_by_ALL(cfg.model, cfg.temploader, tpath, cfg.model.backbone.lower(), cfg.half, save=True)
        return tdict

    if cfg.ttype == 'ALL':
        return get_ALL(cfg, tpath)

    else:
        try:
            tdict = cfg.model.load_template(os.path.join(tpath, tname))

        except:
            tdict = getattr(tl, f'gen_by_{cfg.ttype}')(get_ALL(cfg, tpath), cfg.tsize, tpath, cfg.model.backbone.lower(), num_workers=cfg.num_workers, save=True)

    return tdict

if __name__ == '__main__':
    args = argParse()
    cfg = Cfg(args)
    categories = tqdm(cfg.categories)

    for category in categories:
        torch.cuda.empty_cache()
        categories.set_description(category)
        cfg.update(category)
        globals()[args.mode](cfg)

    if args.mode == 'test':
        cfg.metrics.show()
