"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import sys, os
sys.path.insert(0, os.getcwd())
sys.path.insert(0, '../SBGAN')
#sys.path.insert(0, '../SBGAN/SBGAN')

from collections import OrderedDict
from torch.autograd import Variable

import data
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from util.visualizer import Visualizer
from util import html
from SBGAN.modules.inception import InceptionV3
from SBGAN.modules.fid_score import calculate_fid_given_acts, get_activations


opt = TestOptions().parse()

dataloader = data.create_dataloader(opt)

model = Pix2PixModel(opt)
model.eval()

visualizer = Visualizer(opt)

# create a webpage that summarizes the all results
web_dir = os.path.join(opt.results_dir, opt.name,
                       '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir,
                    'Experiment = %s, Phase = %s, Epoch = %s' %
                    (opt.name, opt.phase, opt.which_epoch))

# test
for i, data_i in enumerate(dataloader):
    if i * opt.batchSize >= opt.how_many:
        break

    generated = model(data_i, mode='inference')

    img_path = data_i['path']
    for b in range(generated.shape[0]):
        print('process image... %s' % img_path[b])
        visuals = OrderedDict([('input_label', data_i['label'][b]),
                               ('synthesized_image', generated[b])])
        visualizer.save_images(webpage, visuals, img_path[b:b + 1])

webpage.save()

compute_FID()

def compute_FID(self):
    nums_fid = 1000
    dims = 2048
    batchsize = 10 
    all_reals = np.zeros((int(nums_fid/batchsize)*batchsize,dims))
    all_fakes = np.zeros((int(nums_fid/batchsize)*batchsize,dims))

    #load inception network
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    self.inception_model = InceptionV3([block_idx])
    self.inception_model.cpu()

    for i, data_i in enumerate(dataloader):
        if i * batchsize >= nums_fid:
            break
        generated = model(data_i, mode='inference')
        fake_acts = get_activations(generated, self.inception_model, batchsize, cuda=True)
        all_fakes[i*batchsize:i*batchsize+fake_acts.shape[0],:] = fake_acts

    for i, data_i in enumerate(dataloader):
        if i * batchsize >= nums_fid:
            break
            real_ims = Variable(data_i['image']).cpu()
            print(img.size())
            real_acts = get_activations(real_ims, self.inception_model, batchsize, cuda=True)
            all_reals[i*batchsize:i*batchsize+real_acts.shape[0],:] = real_acts

    fid_eval = calculate_fid_given_acts(all_reals, all_fakes)
    return fid_eval
