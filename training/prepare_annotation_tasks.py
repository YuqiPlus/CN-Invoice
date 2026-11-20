"""Prepare local annotation tasks for real VAT photos outside the regression set."""
import argparse
import json
import os
from pathlib import Path
from urllib.parse import quote


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if '.private' not in args.output.resolve().parts:parser.error('Private output required')
    os.umask(0o077);args.output.mkdir(parents=True,exist_ok=True)
    references=json.loads((args.source/'frozen-reference.json').read_text())
    heldout={s['id'] for s in references['samples']}
    repeats={s['id'] for s in references['excluded_before_inference'] if 'visual repeat' in s['reason']}
    tasks=[]
    for record in json.loads((args.source/'collection.json').read_text())['samples']:
        if not record['id'].startswith('FUXI-') or record['id'] in heldout|repeats:continue
        tasks.append({'id':len(tasks)+1,'data':{'image':'/data/local-files/?d='+quote(record['file']),
                                            'image_path':str((args.source/record['file']).resolve())},
            'meta':{'sample_id':record['id'],'sha256':record['sha256'],'source_url':record['source_url'],'status':'needs_manual_field_boxes',
                    'instructions':'标数值区域：不含税合计、合计税额、价税合计小写、发票号码。不要标明细；没有字段保持缺失。'}})
    (args.output/'tasks.json').write_text(json.dumps(tasks,ensure_ascii=False,indent=2))
    (args.output/'storage.json').write_text(json.dumps({'local_file_document_root':str(args.source.resolve()),
         'note':'Configure local-file serving explicitly in your annotation tool; this script does not start or configure a service.'},ensure_ascii=False,indent=2))
    (args.output/'label_config.xml').write_text('''<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="field" toName="image">
    <Label value="vat.amount" background="#1976d2"/>
    <Label value="vat.tax" background="#f57c00"/>
    <Label value="vat.total" background="#388e3c"/>
    <Label value="vat.invoice_number" background="#7b1fa2"/>
  </RectangleLabels>
</View>''')
    print(json.dumps({'tasks':len(tasks),'heldout_excluded':True,'suspected_repeat_photos_excluded':len(repeats),
                      'note':'Image paths require a caller-configured local-file storage mapping before Label Studio import; no external service was contacted.'}))


if __name__=='__main__':main()
