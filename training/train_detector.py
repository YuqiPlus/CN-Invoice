"""Fine-tune a compact visual field detector; validation alone selects the checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import Counter
from functools import partial
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
from torchvision.models.detection._utils import retrieve_out_channels
from torchvision.ops import box_iou

from training_policy import category_weights, validation_score, validate_specialist_manifest


def build_model(classes, pretrained=None):
    model=ssdlite320_mobilenet_v3_large(weights=None,weights_backbone=None,num_classes=91)
    if pretrained is not None:
        model.load_state_dict(torch.load(pretrained,map_location='cpu',weights_only=True))
    model.eval()
    channels=retrieve_out_channels(model.backbone,(320,320))
    model.head.classification_head=SSDLiteClassificationHead(channels,model.anchor_generator.num_anchors_per_location(),
        classes,partial(nn.BatchNorm2d,eps=.001,momentum=.03))
    model.anchor_generator.scales=[.08,.14,.25,.4,.6,.8,1.0]
    model.score_thresh=.05
    model.detections_per_img=60
    return model


class FieldDataset(Dataset):
    def __init__(self, root, manifest, split, real_vat_repeat=1,pseudo_repeat=1):
        self.rows=[r for r in manifest['records'] if r['split']==split]
        self.train=split=='train'
        self.items=[]
        for row in self.rows:
            image_path=root/row['file']
            if hashlib.sha256(image_path.read_bytes()).hexdigest()!=row['sha256']:
                raise ValueError('Training image changed after dataset preparation')
            with Image.open(image_path) as image:
                image=image.convert('RGB').resize((320,320),Image.Resampling.BILINEAR)
                pixels=np.array(image,dtype=np.uint8)
            boxes=np.array([a['box'] for a in row['annotations']],dtype=np.float32)
            boxes[:,[0,2]]*=320/row['width'];boxes[:,[1,3]]*=320/row['height']
            labels=[manifest['fields'].index(a['field'])+1 for a in row['annotations']]
            self.items.append((pixels,boxes,np.array(labels,dtype=np.int64)))
            if self.train and row['origin']=='publisher_manual_boxes':
                self.items.extend([(pixels,boxes,np.array(labels,dtype=np.int64))]*(real_vat_repeat-1))
            if self.train and row['origin']=='lite_grounded_pseudo':
                self.items.extend([(pixels,boxes,np.array(labels,dtype=np.int64))]*(pseudo_repeat-1))

    def __len__(self):return len(self.items)

    def __getitem__(self,index):
        pixels,boxes,labels=self.items[index]
        pixels=pixels.copy();boxes=boxes.copy()
        if self.train:
            # Horizontal flips would alter document semantics and are not used.
            if random.random()<.6:
                image=Image.fromarray(pixels)
                image=ImageEnhance.Brightness(image).enhance(random.uniform(.75,1.2))
                image=ImageEnhance.Contrast(image).enhance(random.uniform(.75,1.25))
                pixels=np.array(image)
            if random.random()<.5:
                dx,dy=random.randint(-8,8),random.randint(-8,8)
                image=Image.fromarray(pixels).transform((320,320),Image.Transform.AFFINE,(1,0,-dx,0,1,-dy),fillcolor=(240,240,235))
                pixels=np.array(image);boxes[:,[0,2]]+=dx;boxes[:,[1,3]]+=dy
                boxes=np.clip(boxes,0,320)
        keep=(boxes[:,2]>boxes[:,0]+1)&(boxes[:,3]>boxes[:,1]+1)
        return torch.from_numpy(pixels.copy()).permute(2,0,1).float()/255,{
            'boxes':torch.from_numpy(boxes[keep]),'labels':torch.from_numpy(labels[keep]),'image_id':torch.tensor(index)}


def collate(batch):return tuple(zip(*batch))


@torch.inference_mode()
def validate(model, loader, device, class_count):
    model.eval();correct=total=false_positive=0;by_class={i:[0,0] for i in range(1,class_count)};unmatched_by_class=Counter()
    for images,targets in loader:
        predictions=model([image.to(device) for image in images])
        for prediction,target in zip(predictions,targets):
            for label, box in zip(target['labels'],target['boxes']):
                cls=int(label);total+=1;by_class[cls][1]+=1
                mask=(prediction['labels']==cls)&(prediction['scores']>=.35)
                selected=prediction['boxes'][mask].cpu()
                match=bool(len(selected) and box_iou(box[None],selected).max()>=.5)
                correct+=match;by_class[cls][0]+=match
            for box,label,score in zip(prediction['boxes'].cpu(),prediction['labels'].cpu(),prediction['scores'].cpu()):
                if score<.35:continue
                truth=target['boxes'][target['labels']==label]
                if not len(truth) or box_iou(box[None],truth).max()<.5:
                    false_positive+=1;unmatched_by_class[int(label)]+=1
    return {'matched_fields_iou50':correct,'fields':total,'recall_iou50':correct/max(total,1),
            'unmatched_detections':false_positive,'by_class':by_class,'unmatched_by_class':dict(unmatched_by_class)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--pretrained',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=12)
    parser.add_argument('--batch-size',type=int,default=24)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--real-vat-repeat',type=int,default=1)
    parser.add_argument('--pseudo-repeat',type=int,default=1)
    parser.add_argument('--lr-milestones',type=int,nargs=2,default=[8,11])
    parser.add_argument('--category-balance',choices=['none','sqrt'],default='none')
    parser.add_argument('--selection-metric',choices=['micro','macro-category'],default='micro')
    args=parser.parse_args()
    if '.private' not in args.output.resolve().parts:parser.error('Private checkpoint output required')
    args.output.mkdir(parents=True,exist_ok=True)
    random.seed(20261119);np.random.seed(20261119);torch.manual_seed(20261119);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True
    manifest=json.loads(args.dataset.read_text());device=torch.device(args.device)
    validate_specialist_manifest(manifest)
    if not 1<=args.real_vat_repeat<=100:parser.error('real-vat-repeat must be between 1 and 100')
    if not 1<=args.pseudo_repeat<=20:parser.error('pseudo-repeat must be between 1 and 20')
    datasets={s:FieldDataset(args.dataset.parent,manifest,s,args.real_vat_repeat,args.pseudo_repeat) for s in ('train','val')}
    sampler=None
    if args.category_balance=='sqrt':
        if args.real_vat_repeat!=1 or args.pseudo_repeat!=1:parser.error('Category balancing requires repeat factors of 1')
        sampler=WeightedRandomSampler(category_weights([r['category'] for r in datasets['train'].rows]),len(datasets['train']),replacement=True)
    loaders={s:DataLoader(datasets[s],batch_size=args.batch_size,shuffle=s=='train' and sampler is None,
                          sampler=sampler if s=='train' else None,num_workers=0,collate_fn=collate,
                          drop_last=s=='train',pin_memory=device.type=='cuda') for s in ('train','val')}
    model=build_model(len(manifest['fields'])+1,args.pretrained).to(device)
    optimizer=torch.optim.SGD(model.parameters(),lr=.004,momentum=.9,weight_decay=.0005)
    if not 0 < args.lr_milestones[0] < args.lr_milestones[1]:parser.error('Learning-rate milestones must be positive and increasing')
    schedule=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=args.lr_milestones,gamma=.2)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    best=-float('inf');history=[];started=time.perf_counter()
    for epoch in range(args.epochs):
        model.train();losses=[];tick=time.perf_counter()
        for images,targets in loaders['train']:
            images=[image.to(device) for image in images]
            targets=[{k:v.to(device) for k,v in target.items()} for target in targets]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type,enabled=device.type=='cuda'):
                loss=sum(model(images,targets).values())
            if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss')
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(),10)
            scaler.step(optimizer);scaler.update();losses.append(float(loss.detach()))
        validation=validate(model,loaders['val'],device,len(manifest['fields'])+1)
        metric=validation_score(validation,manifest['fields'],args.selection_metric)
        row={'epoch':epoch+1,'train_loss':sum(losses)/len(losses),'validation':validation,'selection_score':metric,'seconds':round(time.perf_counter()-tick,3)}
        history.append(row)
        if metric>best:
            best=metric
            torch.save({'state_dict':model.cpu().state_dict(),'fields':manifest['fields'],'epoch':epoch+1,
                        'dataset_sha256':hashlib.sha256(args.dataset.read_bytes()).hexdigest()},args.output/'best.pt')
            model.to(device)
        schedule.step()
        (args.output/'history.json').write_text(json.dumps(history,indent=2))
        print(json.dumps({'epoch':epoch+1,'loss':round(row['train_loss'],4),'val_recall_iou50':round(validation['recall_iou50'],4),
                          'val_unmatched':validation['unmatched_detections'],'seconds':row['seconds']}),flush=True)
    metadata={'architecture':'ssdlite320_mobilenet_v3_large','fields':manifest['fields'],'input_size':320,'score_threshold':.35,
        'document_profile':manifest.get('document_profile'),
        'parameters':sum(p.numel() for p in model.parameters()),'epochs':args.epochs,'best_validation_score':best,
        'training_seconds':round(time.perf_counter()-started,3),'data_counts':manifest['counts'],
        'real_vat_repeat':args.real_vat_repeat,'effective_train_exposures_per_epoch':len(datasets['train']),
        'pseudo_repeat':args.pseudo_repeat,'pseudo_summary':manifest.get('pseudo_summary'),
        'lr_milestones':args.lr_milestones,
        'category_balance':args.category_balance,'selection_metric':args.selection_metric,
        'training_field_counts':dict(Counter(a['field'] for r in manifest['records'] if r['split']=='train' for a in r['annotations'])),
        'training_origins':dict(Counter(r['origin'] for r in manifest['records'] if r['split']=='train')),
        'dataset_sha256':hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        'pretrained_sha256':hashlib.sha256(args.pretrained.read_bytes()).hexdigest(),
        'training_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'limitations':['Experimental field detector; see training_origins and training_field_counts for actual supervision. Repeated exposures are not new invoices.',
                       'Lite-generated labels are not human audited truth; same-teacher validation does not establish business accuracy.',
                       'Validation selects the checkpoint; independent reference tests are not consulted during training.']}
    (args.output/'model.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata),flush=True)


if __name__=='__main__':main()
