"""Collect explicitly discovered public sample media, preserving provenance and holdout isolation."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
from PIL import Image, ImageOps


def digest(data):return hashlib.sha256(data).hexdigest()
def pixels(image):return digest(str(image.size).encode()+image.convert('RGB').tobytes())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--discovered',type=Path,required=True)
    parser.add_argument('--extra-urls',type=Path)
    parser.add_argument('--heldout',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if '.private' not in args.output.resolve().parts:parser.error('Private output required')
    os.umask(0o077);args.output.mkdir(parents=True,exist_ok=True)
    documents=args.output/'documents';documents.mkdir(exist_ok=True)
    images=args.output/'images';images.mkdir(exist_ok=True)
    rows=[f for repo in json.loads(args.discovered.read_text()) for f in repo['files']]
    if args.extra_urls:rows.extend(json.loads(args.extra_urls.read_text()))
    unique={r['url']:r for r in rows};rows=list(unique.values())
    heldout=json.loads(args.heldout.read_text())['samples'];forbidden_hashes={r['sha256'] for r in heldout}
    forbidden_pixels={r['pixel_sha256'] for r in heldout}
    records=[];errors=[];seen=set();documents_ok=0
    def fetch(row):
        if row.get('local_file'):
            path = Path(row['local_file']).resolve()
            if '.private' not in path.parts or not row.get('sha256'):
                return row,path,'InvalidLocalSource'
            if digest(path.read_bytes()) != row['sha256']:
                return row,path,'LocalSourceHashChanged'
            return row,path,None
        suffix='.pdf' if row['url'].lower().split('?')[0].endswith('.pdf') else Path(row.get('path','image.jpg')).suffix.lower()
        target=documents/(digest(row['url'].encode())[:20]+suffix)
        try:
            if not target.exists():
                with httpx.Client(timeout=45,follow_redirects=True) as client:
                    with client.stream('GET',row['url']) as response:
                        response.raise_for_status();data=bytearray()
                        for block in response.iter_bytes():
                            data.extend(block)
                            if len(data)>25*1024**2:raise ValueError('Source media exceeds download size limit')
                target.write_bytes(data)
            return row,target,None
        except Exception as exc:return row,target,type(exc).__name__
    fetched=[]
    with ThreadPoolExecutor(max_workers=6) as pool:
        for future in as_completed([pool.submit(fetch,row) for row in rows]):fetched.append(future.result())
    def save(image,row,**extra):
        image=ImageOps.exif_transpose(image).convert('RGB')
        if min(image.size)<180 or max(image.size)<400:return
        pixel=pixels(image)
        if pixel in forbidden_pixels or pixel in seen:return
        seen.add(pixel)
        image.thumbnail((4000,4000),Image.Resampling.LANCZOS)
        identifier='WEB-'+pixel[:20];path=images/(identifier+'.png');image.save(path)
        records.append({'id':identifier,'file':str(path.resolve()),'sha256':digest(path.read_bytes()),'pixel_sha256':pixels(image),
            'width':image.width,'height':image.height,'source_url':row['url'],'source_repo':row.get('repo'),
            'source_license':row.get('license'),'source_realness':'unverified_public_sample',**extra})
    for row,path,error in fetched:
        if error:errors.append({'url':row['url'],'error':error});continue
        raw=path.read_bytes()
        if digest(raw) in forbidden_hashes:continue
        try:
            if raw.startswith(b'%PDF'):
                import pypdfium2 as pdfium
                from cninvoice.parsers import parse_text
                from cninvoice.models import Source
                document=pdfium.PdfDocument(raw);pages=0
                for number in range(min(len(document),160)):
                    page=document[number];textpage=page.get_textpage();text=textpage.get_text_range();textpage.close()
                    for obj in page.get_objects(filter=[pdfium.raw.FPDF_PAGEOBJ_IMAGE]):
                        try:
                            embedded=obj.get_bitmap();part=embedded.to_pil().copy()
                            if part.width>=500 and part.height>=250:
                                save(part,row,page=number+1,extraction='embedded_image')
                            # get_bitmap() is a borrowed PDFium bitmap; its owner closes it.
                            part.close()
                        except Exception:pass
                    if len(document)>3 and not ('价税合计' in text and ('发票' in text)):
                        page.close();continue
                    scale=min(2000/page.get_width(),2000/page.get_height())
                    bitmap=page.render(scale=scale);image=bitmap.to_pil().copy()
                    parsed=parse_text(text,Source.PDF_TEXT)
                    native={k:str(getattr(parsed,k)) for k in ('amount','tax','total','invoice_number') if getattr(parsed,k) is not None}
                    save(image,row,page=number+1,native_values=native,native_ambiguous=bool(parsed.issues))
                    image.close();bitmap.close();page.close();pages+=1
                    if pages>=16:break
                document.close()
            else:
                with Image.open(path) as image:save(image,row)
            documents_ok+=1
        except Exception as exc:errors.append({'url':row['url'],'error':type(exc).__name__})
    result={'status':'complete','discovered_urls':len(rows),'downloaded_and_decoded_documents':documents_ok,
            'samples':records,'errors':errors,'notes':['These are candidates, not verified Chinese VAT invoices.',
            'Screenshots, empty samples and non-invoices are filtered by the teacher/gates before training.',
            'Source media is private and not redistributed in the package.']}
    (args.output/'collection.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({k:result[k] for k in ('status','discovered_urls','downloaded_and_decoded_documents')}|{'image_candidates':len(records),'errors':len(errors)}))


if __name__=='__main__':main()
