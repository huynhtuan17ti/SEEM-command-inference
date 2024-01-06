import torch
import argparse
import os
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from utils.visualizer import Visualizer
from modeling.BaseModel import BaseModel
from modeling import build_model
import cv2
from utils.distributed import init_distributed
from utils.arguments import load_opt_from_config_files
from utils.constants import COCO_PANOPTIC_CLASSES
from detectron2.data import MetadataCatalog
from modeling.language.loss import vl_similarity
from utils.constants import COCO_PANOPTIC_CLASSES
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES

from PIL import Image

def parse_option():
    parser = argparse.ArgumentParser('Reference image inference', add_help=False)
    parser.add_argument('--conf_files', default="configs/seem/focall_unicl_lang_demo.yaml", metavar="FILE", help='path to config file', )
    cfg = parser.parse_args()
    return cfg

def main():
    # currently, only work with seem demo
    cfg = parse_option()
    opt = load_opt_from_config_files([cfg.conf_files])
    opt = init_distributed(opt)

    # META DATA
    pretrained_pth = "model_weights/seem_focall_v0.pt"
    refimg_pth = "test_images/ref_set/ref_clown.jpg"
    refmask_pth = "test_images/ref_set/mask_clown.png"
    image_dir = "test_images/ref_set/clown/"
    output_dir = "results/clown/"
    reftxt = "clown"

    model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()
    with torch.no_grad():
        model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(COCO_PANOPTIC_CLASSES + ["background"], is_eval=True)

    # inistalize task
    model.model.task_switch['spatial'] = False
    model.model.task_switch['visual'] = False
    model.model.task_switch['grounding'] = False
    model.model.task_switch['audio'] = False

    # prepare colors and transform
    t = []
    t.append(transforms.Resize(512, interpolation=Image.BICUBIC))
    transform = transforms.Compose(t)
    metadata = MetadataCatalog.get('coco_2017_train_panoptic')
    all_classes = [name.replace('-other','').replace('-merged','') for name in COCO_PANOPTIC_CLASSES] + ["others"]
    colors_list = [(np.array(color['color'])/255).tolist() for color in COCO_CATEGORIES] + [[1, 1, 1]]

    # inference main

    @torch.no_grad()
    def handle_refimg():
        model.model.task_switch['visual'] = True
        model.model.task_switch['spatial'] = True
        refimg_ori = Image.open(refimg_pth)
        # check if refmask_pth is a npy or image
        if refmask_pth.endswith("npy"):
            refimg_mask = np.load(refmask_pth)
            # 1-channel to 3-channel
            refimg_mask = cv2.merge((refimg_mask, refimg_mask, refimg_mask))
        else:
            refimg_mask = Image.open(refmask_pth).convert("RGB")
            refimg_mask = np.asarray(refimg_mask)
        
        refimg_ori = transform(refimg_ori)
        _width = refimg_ori.size[0]
        _height = refimg_ori.size[1]
        refimg_ori = np.asarray(refimg_ori)
        images = torch.from_numpy(refimg_ori.copy()).permute(2,0,1).cuda()
        batched_inputs = [{'image': images, 'height': _height, 'width': _width, 'spatial_query':{}}]

        refimg_mask = refimg_mask[:,:,0:1].copy()
        refimg_mask = torch.from_numpy(refimg_mask).permute(2,0,1)[None,]
        refimg_mask = (F.interpolate(refimg_mask, (_height, _width), mode='bilinear') > 0)
        batched_inputs[0]['spatial_query']['rand_shape'] = refimg_mask
        outputs_refimg, img_shape = model.model.evaluate_referring_image(batched_inputs)
        model.model.task_switch['spatial'] = False

        return outputs_refimg
    
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        outputs_refimg = handle_refimg()

    @torch.no_grad()
    def inference_single_image(image_pth):
        image_ori = transform(Image.open(image_pth).convert('RGB'))
        width = image_ori.size[0]
        height = image_ori.size[1]
        image_ori = np.asarray(image_ori)
        visual = Visualizer(image_ori, metadata=metadata)
        images = torch.from_numpy(image_ori.copy()).permute(2,0,1).cuda()

        data = {"image": images, "height": height, "width": width}

        # handle refimg
        data["visual"] =  outputs_refimg

        # handle text
        model.model.task_switch['grounding'] = True
        data['text'] = [reftxt]

        batch_inputs = [data]
        results, image_size, extra = model.model.evaluate_demo(batch_inputs)

        v_emb = results['pred_maskembs']
        s_emb = results['pred_pvisuals']
        pred_masks = results['pred_masks']

        pred_logits = v_emb @ s_emb.transpose(1,2)
        logits_idx_y = pred_logits[:,:,0].max(dim=1)[1]
        logits_idx_x = torch.arange(len(logits_idx_y), device=logits_idx_y.device)
        logits_idx = torch.stack([logits_idx_x, logits_idx_y]).tolist()
        pred_masks_pos = pred_masks[logits_idx]
        pred_class = results['pred_logits'][logits_idx].max(dim=-1)[1]

        pred_masks_pos = (F.interpolate(pred_masks_pos[None,], image_size[-2:], mode='bilinear')[0,:,:data['height'],:data['width']] > 0.0).float().cpu().numpy()
        texts = [all_classes[pred_class[0]]]

        for idx, mask in enumerate(pred_masks_pos):
            # color = random_color(rgb=True, maximum=1).astype(np.int32).tolist()
            out_txt = reftxt
            demo = visual.draw_binary_mask(mask, color=colors_list[pred_class[0]%133], text=out_txt)
        res = demo.get_image()
        torch.cuda.empty_cache()
        # return Image.fromarray(res), stroke_inimg, stroke_refimg
        return Image.fromarray(res)

    for image_file in os.listdir(image_dir):
        image_name = image_file.split('.')[0]
        image_pth = os.path.join(image_dir, image_file)
        print(f"Processing {image_pth} ....", end = " ")
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            output_img = inference_single_image(image_pth)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_img.save(os.path.join(output_dir, f'{image_name}.png'))
        print('Done!')

if __name__ == '__main__':
    main()