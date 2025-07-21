import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'VITON-HD'))
import torch
from torch import nn
from torch.nn import functional as F
import torchgeometry as tgm

from datasets import VITONDataset, VITONDataLoader
from networks import SegGenerator, GMM, ALIASGenerator
from utils import gen_noise, load_checkpoint, save_images

class VitonHDOptions:
    def __init__(
        self,
        name="viton_hd",
        batch_size=1,
        workers=1,
        load_height=1024,
        load_width=768,
        shuffle=False,
        dataset_dir="./VITON-HD/datasets/",
        dataset_mode="test",
        dataset_list="test_pairs.txt",
        checkpoint_dir="./VITON-HD/checkpoints/",
        save_dir="./VITON-HD/results/",
        display_freq=1,
        seg_checkpoint="seg_final.pth",
        gmm_checkpoint="gmm_final.pth",
        alias_checkpoint="alias_final.pth",
        semantic_nc=13,
        init_type="xavier",
        init_variance=0.02,
        grid_size=5,
        norm_G="spectralaliasinstance",
        ngf=64,
        num_upsampling_layers="most"
    ):
        self.name = name
        self.batch_size = batch_size
        self.workers = workers
        self.load_height = load_height
        self.load_width = load_width
        self.shuffle = shuffle
        self.dataset_dir = dataset_dir
        self.dataset_mode = dataset_mode
        self.dataset_list = dataset_list
        self.checkpoint_dir = checkpoint_dir
        self.save_dir = save_dir
        self.display_freq = display_freq
        self.seg_checkpoint = seg_checkpoint
        self.gmm_checkpoint = gmm_checkpoint
        self.alias_checkpoint = alias_checkpoint
        self.semantic_nc = semantic_nc
        self.init_type = init_type
        self.init_variance = init_variance
        self.grid_size = grid_size
        self.norm_G = norm_G
        self.ngf = ngf
        self.num_upsampling_layers = num_upsampling_layers

def run_viton_hd(opt: VitonHDOptions):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.exists(os.path.join(opt.save_dir, opt.name)):
        os.makedirs(os.path.join(opt.save_dir, opt.name))

    seg = SegGenerator(opt, input_nc=opt.semantic_nc + 8, output_nc=opt.semantic_nc)
    gmm = GMM(opt, inputA_nc=7, inputB_nc=3)
    opt.semantic_nc = 7
    alias = ALIASGenerator(opt, input_nc=9)
    opt.semantic_nc = 13

    load_checkpoint(seg, os.path.join(opt.checkpoint_dir, opt.seg_checkpoint))
    load_checkpoint(gmm, os.path.join(opt.checkpoint_dir, opt.gmm_checkpoint))
    load_checkpoint(alias, os.path.join(opt.checkpoint_dir, opt.alias_checkpoint))

    seg = seg.to(device).eval()
    gmm = gmm.to(device).eval()
    alias = alias.to(device).eval()

    up = nn.Upsample(size=(opt.load_height, opt.load_width), mode='bilinear').to(device)
    gauss = tgm.image.GaussianBlur((15, 15), (3, 3)).to(device)

    test_dataset = VITONDataset(opt)
    test_loader = VITONDataLoader(opt, test_dataset)

    result_files = []
    with torch.no_grad():
        for i, inputs in enumerate(test_loader.data_loader):
            img_names = inputs['img_name']
            c_names = inputs['c_name']['unpaired']

            img_agnostic = inputs['img_agnostic'].to(device)
            parse_agnostic = inputs['parse_agnostic'].to(device)
            pose = inputs['pose'].to(device)
            c = inputs['cloth']['unpaired'].to(device)
            cm = inputs['cloth_mask']['unpaired'].to(device)

            # Part 1. Segmentation generation
            parse_agnostic_down = F.interpolate(parse_agnostic, size=(256, 192), mode='bilinear')
            pose_down = F.interpolate(pose, size=(256, 192), mode='bilinear')
            c_masked_down = F.interpolate(c * cm, size=(256, 192), mode='bilinear')
            cm_down = F.interpolate(cm, size=(256, 192), mode='bilinear')
            seg_input = torch.cat((cm_down, c_masked_down, parse_agnostic_down, pose_down, gen_noise(cm_down.size()).to(device)), dim=1)

            parse_pred_down = seg(seg_input)
            parse_pred = gauss(up(parse_pred_down))
            parse_pred = parse_pred.argmax(dim=1)[:, None]

            parse_old = torch.zeros(parse_pred.size(0), 13, opt.load_height, opt.load_width, dtype=torch.float, device=device)
            parse_old.scatter_(1, parse_pred, 1.0)

            labels = {
                0:  ['background',  [0]],
                1:  ['paste',       [2, 4, 7, 8, 9, 10, 11]],
                2:  ['upper',       [3]],
                3:  ['hair',        [1]],
                4:  ['left_arm',    [5]],
                5:  ['right_arm',   [6]],
                6:  ['noise',       [12]]
            }
            parse = torch.zeros(parse_pred.size(0), 7, opt.load_height, opt.load_width, dtype=torch.float, device=device)
            for j in range(len(labels)):
                for label in labels[j][1]:
                    parse[:, j] += parse_old[:, label]

            # Part 2. Clothes Deformation
            agnostic_gmm = F.interpolate(img_agnostic, size=(256, 192), mode='nearest')
            parse_cloth_gmm = F.interpolate(parse[:, 2:3], size=(256, 192), mode='nearest')
            pose_gmm = F.interpolate(pose, size=(256, 192), mode='nearest')
            c_gmm = F.interpolate(c, size=(256, 192), mode='nearest')
            gmm_input = torch.cat((parse_cloth_gmm, pose_gmm, agnostic_gmm), dim=1)

            _, warped_grid = gmm(gmm_input, c_gmm)
            warped_c = F.grid_sample(c, warped_grid, padding_mode='border')
            warped_cm = F.grid_sample(cm, warped_grid, padding_mode='border')

            # Part 3. Try-on synthesis
            misalign_mask = parse[:, 2:3] - warped_cm
            misalign_mask[misalign_mask < 0.0] = 0.0
            parse_div = torch.cat((parse, misalign_mask), dim=1)
            parse_div[:, 2:3] -= misalign_mask

            output = alias(torch.cat((img_agnostic, pose, warped_c), dim=1), parse, parse_div, misalign_mask)

            unpaired_names = []
            for img_name, c_name in zip(img_names, c_names):
                unpaired_names.append('{}_{}'.format(img_name.split('_')[0], c_name))
            save_images(output, unpaired_names, os.path.join(opt.save_dir, opt.name))
            result_files.extend([
                os.path.join(opt.save_dir, opt.name, f"{name}.png") for name in unpaired_names
            ])

            if (i + 1) % opt.display_freq == 0:
                print("step: {}".format(i + 1))
    return result_files

# Example usage:
if __name__ == "__main__":
    opt = VitonHDOptions(
        name="single_test",
        dataset_dir="./VITON-HD/datasets/single_test/",
        checkpoint_dir=os.path.join(os.path.dirname(__file__), 'VITON-HD/checkpoints/'),
        save_dir=os.path.join(os.path.dirname(__file__), 'VITON-HD/results/')
    )
    result_files = run_viton_hd(opt)
    print("Generated files:", result_files)