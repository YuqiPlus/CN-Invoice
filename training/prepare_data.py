"""Prepare a private pilot set. The established 125-image regression set is excluded."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import re
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

from PIL import Image, ImageDraw, ImageFont

FIELDS = ['vat.amount', 'vat.tax', 'vat.total', 'vat.invoice_number',
          'rail.ticket_number', 'rail.departure_station', 'rail.train_number',
          'rail.arrival_station', 'rail.departure', 'rail.total', 'rail.seat_class', 'rail.passenger_name']
PICK_FIELDS = dict(zip(['ticket_num', 'starting_station', 'train_num', 'destination_station', 'date',
                       'ticket_rates', 'seat_category', 'name'], FIELDS[4:]))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def pixel_hash(image):
    return sha(str(image.size).encode() + image.convert('RGB').tobytes())


def parse_pick(text):
    result = []
    for row in csv.reader(io.StringIO(text)):
        if len(row) < 11:
            continue
        try:
            coords = list(map(float, row[1:9]))
        except ValueError:
            continue
        label = row[-1].strip()
        if label in PICK_FIELDS:
            result.append({'field': PICK_FIELDS[label], 'text': ','.join(row[9:-1]).strip(),
                           'box': [min(coords[::2]), min(coords[1::2]), max(coords[::2]), max(coords[1::2])]})
    return result


def group_key(annotations, fallback):
    ticket = next((a['text'] for a in annotations if a['field'] == 'rail.ticket_number'), None)
    return 'pick-ticket:' + re.sub(r'\s', '', ticket) if ticket else 'pick-image:' + fallback


def split_group(key):
    return 'val' if int(sha(key.encode())[:8], 16) % 10 == 0 else 'train'


def synthetic_vat(index, rng, font_path):
    width, height = 1280, 820
    paper = rng.choice([(250, 249, 242), (239, 239, 228), (255, 255, 255)])
    image = Image.new('RGB', (width, height), paper)
    draw = ImageDraw.Draw(image)
    ink = rng.choice([(30, 38, 100), (20, 20, 20), (35, 54, 80)])
    grid = rng.choice([(50, 110, 75), (140, 55, 50), (65, 65, 65)])
    fonts = {size: ImageFont.truetype(font_path, size) for size in (18, 22, 26, 30, 34)}
    offset_x, offset_y = rng.randint(-18, 18), rng.randint(-30, 20)
    left, right = 45 + offset_x, 1230 + offset_x
    top, bottom = 230 + offset_y, 600 + offset_y
    draw.text((width // 2, 30), rng.choice(['增值税专用发票', '电子发票（普通发票）', '电子发票（增值税专用发票）']),
              anchor='mt', font=fonts[34], fill=grid)
    draw.text((left, 110), '购买方：示例采购企业', font=fonts[22], fill=ink)
    draw.text((left, 160), '销售方：示例销售企业', font=fonts[22], fill=ink)
    draw.text((820, 160), '开票日期：2026年11月19日', font=fonts[18], fill=ink)
    annotations = []
    def value(field, text, x, y, size=26):
        bounds = draw.textbbox((x, y), text, font=fonts[size])
        draw.text((x, y), text, font=fonts[size], fill=ink)
        annotations.append({'field': field, 'text': text,
                            'box': [bounds[0]-3, bounds[1]-3, bounds[2]+3, bounds[3]+3]})
    number = ''.join(str(rng.randrange(10)) for _ in range(rng.choice([8, 20])))
    draw.text((760 if len(number) == 20 else 925, 90), '发票号码', font=fonts[18], fill=grid)
    value('vat.invoice_number', number, 760 if len(number) == 20 else 925, 115, 22)
    amount_cents = rng.randrange(100, 5_000_000)
    rate = rng.choice([0, 1, 3, 6, 9, 13])
    tax_cents = (amount_cents * rate + 50) // 100
    total_cents = amount_cents + tax_cents
    money = lambda cents: f'{cents // 100}.{cents % 100:02d}'
    cols = [left, left+400, left+540, left+680, left+865, left+970, right]
    draw.rectangle((left, top, right, bottom), outline=grid, width=2)
    for x in cols[1:-1]:
        draw.line((x, top, x, bottom), fill=grid, width=2)
    draw.line((left, top+50, right, top+50), fill=grid, width=2)
    for x, label in zip(cols, ['货物或服务名称', '数量', '单价', '金额', '税率', '税额']):
        draw.text((x+8, top+10), label, fill=grid, font=fonts[22])
    count = rng.choice([1, 2, 3, 4])
    parts = [amount_cents // count] * count
    parts[-1] += amount_cents - sum(parts)
    for j, cents in enumerate(parts):
        y = top + 85 + j * 48
        draw.text((left+12, y), f'*服务*示例项目{j+1}', font=fonts[22], fill=ink)
        for x, text in [(cols[1]+15, str(rng.randrange(1, 8))), (cols[2]+5, money(cents)),
                        (cols[3]+8, money(cents)), (cols[4]+10, f'{rate}%'),
                        (cols[5]+8, money((cents * rate + 50) // 100))]:
            draw.text((x, y), text, font=fonts[22], fill=ink)
    sum_y = bottom - 48
    draw.text((left+60, sum_y), '合 计', fill=grid, font=fonts[26])
    value('vat.amount', '￥' + money(amount_cents), cols[3]+5, sum_y)
    value('vat.tax', '￥' + money(tax_cents), cols[5]+5, sum_y)
    gross_y = bottom + rng.randrange(20, 38)
    draw.rectangle((left, bottom, right, gross_y+58), outline=grid, width=2)
    draw.text((left+10, gross_y), '价税合计（大写）', font=fonts[22], fill=grid)
    draw.text((left+420, gross_y), '（合成训练票样）', font=fonts[22], fill=ink)
    draw.text((right-365, gross_y), '（小写）', font=fonts[22], fill=grid)
    value('vat.total', '￥' + money(total_cents), right-260, gross_y, 30)
    draw.text((left, height-55), '此图为随机生成的训练样本，不可报销', font=fonts[18], fill=(130,130,130))
    if rng.random() < .7:
        x, y = rng.randrange(800, 1040), rng.randrange(600, 675)
        draw.ellipse((x, y, x+145, y+95), outline=(180,65,65), width=3)
        draw.text((x+12,y+25),'示例专用章',font=fonts[18],fill=(180,65,65))
    return image, annotations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--font', type=Path, required=True)
    parser.add_argument('--synthetic-vat', type=int, default=1000)
    parser.add_argument('--include-nonheldout-paddle',action='store_true')
    args = parser.parse_args()
    if '.private' not in args.output.resolve().parts:
        parser.error('Private training output required')
    os.umask(0o077)
    args.output.mkdir(parents=True, exist_ok=True)
    images = args.output / 'images';images.mkdir(exist_ok=True)
    references = json.loads((args.source / 'frozen-reference.json').read_text())
    forbidden_bytes = {s['sha256'] for s in references['samples']}
    forbidden_pixels = {s['pixel_sha256'] for s in references['samples']}
    forbidden_groups = set()
    collection = json.loads((args.source / 'collection.json').read_text())
    for sample in collection['samples']:
        if sample['id'].startswith('PICK-REAL'):
            number = sample['reference_entities'].get('ticket_num', [])
            if number:forbidden_groups.add('pick-ticket:' + re.sub(r'\s', '', number[0]))
    records, exclusions = [], Counter()
    seen_pixels = set()
    with ZipFile(args.source / 'pick-tickets.zip') as archive:
        rows = list(csv.reader(io.StringIO(archive.read('train_tickets/synth1530_real320_baseline_trainset.csv').decode())))
        for row in rows:
            stem = row[2]
            data = archive.read('train_tickets/images1930/' + stem + '.jpg')
            with Image.open(io.BytesIO(data)) as image:
                width, height = image.size
                pixels = pixel_hash(image)
            annotations = parse_pick(archive.read('train_tickets/boxes_trans/' + stem + '.tsv').decode())
            key = group_key(annotations, stem)
            reason = ('heldout_bytes' if sha(data) in forbidden_bytes else 'heldout_pixels' if pixels in forbidden_pixels
                      else 'heldout_business_group' if key in forbidden_groups else 'duplicate_pixels' if pixels in seen_pixels else None)
            if reason:
                exclusions[reason] += 1;continue
            seen_pixels.add(pixels)
            for item in annotations:
                x1,y1,x2,y2=item['box'];item['box']=[max(0,x1),max(0,y1),min(width,x2),min(height,y2)]
            annotations=[a for a in annotations if a['box'][2]>a['box'][0] and a['box'][3]>a['box'][1]]
            if not annotations:
                exclusions['empty_annotations'] += 1;continue
            file=images/('PICK-TRAIN-'+stem+'.jpg');file.write_bytes(data)
            records.append({'id':'PICK-TRAIN-'+stem,'file':str(file.relative_to(args.output)), 'width':width,'height':height,
                'sha256':sha(data),'pixel_sha256':pixels,'group':sha(key.encode()),'split':split_group(key),'category':'rail',
                'origin':'publisher_train_mixed_real_and_synthetic','annotations':annotations})
    if args.include_nonheldout_paddle:
        tasks=json.loads((args.source/'paddle-annotations.json').read_text())
        heldout_ids={s['id'] for s in references['samples']}
        for sample in collection['samples']:
            if not sample['id'].startswith('PADDLE-') or sample['id'] in heldout_ids:continue
            if sample['sha256'] in forbidden_bytes or sample['pixel_sha256'] in forbidden_pixels:continue
            stem=sample['id'].split('-',1)[1]
            task=next(t for t in tasks if Path(t['data']['image']).stem.split('-',1)[-1]==stem)
            mapping={'金额':'vat.amount','税额':'vat.tax','价税合计':'vat.total','No':'vat.invoice_number'}
            annotations=[]
            for annotation in task['annotations']:
                for item in annotation['result']:
                    value=item['value'];labels=value.get('rectanglelabels',[])
                    if len(labels)!=1 or labels[0] not in mapping:continue
                    if value.get('rotation',0)!=0:raise ValueError('Rotated source boxes need manual conversion')
                    x,y,w,h=[value[k]/100 for k in ('x','y','width','height')]
                    annotations.append({'field':mapping[labels[0]],'box':[x*sample['width'],y*sample['height'],(x+w)*sample['width'],(y+h)*sample['height']]})
            if {a['field'] for a in annotations}!=set(FIELDS[:4]):raise ValueError('Incomplete real VAT annotations')
            data=(args.source/sample['file']).read_bytes();file=images/(sample['id']+'.jpg');file.write_bytes(data)
            records.append({'id':sample['id'],'file':str(file.relative_to(args.output)),'width':sample['width'],'height':sample['height'],
                'sha256':sample['sha256'],'pixel_sha256':sample['pixel_sha256'],'group':sha(sample['id'].encode()),'split':'train',
                'category':'vat','origin':'publisher_manual_boxes','annotations':annotations})
    rng=random.Random(20261119)
    for index in range(args.synthetic_vat):
        image,annotations=synthetic_vat(index,rng,str(args.font));file=images/f'SYNTH-VAT-{index:05}.jpg'
        image.save(file,quality=rng.randrange(65,96))
        records.append({'id':file.stem,'file':str(file.relative_to(args.output)),'width':image.width,'height':image.height,
            'sha256':sha(file.read_bytes()),'group':sha(file.stem.encode()),'split':'val' if index%10==0 else 'train',
            'category':'vat','origin':'synthetic_not_real_invoice','annotations':annotations})
    assert not ({r['group'] for r in records if r['split']=='train'} & {r['group'] for r in records if r['split']=='val'})
    manifest={'version':1,'fields':FIELDS,'records':records,'reference_sha256':sha((args.source/'frozen-reference.json').read_bytes()),
        'exclusions':dict(exclusions),'counts':dict(Counter(r['split']+':'+r['category'] for r in records)),
        'notes':['PICK publisher training split is mixed synthetic/real, not 1850 real tickets.',
                 'VAT training is synthetic, with at most three real non-heldout Paddle images when explicitly enabled; more real annotation is required.',
                 'Original 125 regression images, exact pixels and known test ticket business IDs excluded.',
                 'No airline or filled digital-invoice training data; visual regression set was already used for earlier diagnostics.']}
    (args.output/'dataset.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(json.dumps({'records':len(records),'counts':manifest['counts'],'exclusions':dict(exclusions)},ensure_ascii=False))


if __name__=='__main__':main()
