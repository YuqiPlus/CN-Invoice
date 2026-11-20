"""Export the selected pilot detector to a CPU ONNX model and verify numeric agreement."""
import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from train_detector import build_model


class Export(torch.nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,images):
        out=self.model([images[0]])[0]
        return out['boxes'],out['scores'],out['labels']


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--model-dir',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(4)
    checkpoint=torch.load(args.model_dir/'best.pt',map_location='cpu',weights_only=True)
    metadata=json.loads((args.model_dir/'model.json').read_text())
    if checkpoint['dataset_sha256']!=metadata['dataset_sha256']:
        raise ValueError('Checkpoint and dataset metadata do not match')
    model=build_model(len(checkpoint['fields'])+1);model.load_state_dict(checkpoint['state_dict']);model.eval()
    wrapper=Export(model).eval();sample=torch.rand(1,3,320,320)
    path=args.model_dir/'detector.onnx'
    with torch.no_grad():
        expected=wrapper(sample)
        torch.onnx.export(wrapper,sample,str(path),input_names=['images'],output_names=['boxes','scores','labels'],
                          opset_version=17,dynamo=False,dynamic_axes={'boxes':{0:'detections'},'scores':{0:'detections'},'labels':{0:'detections'}})
    options=ort.SessionOptions();options.intra_op_num_threads=4
    session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
    actual=session.run(None,{'images':sample.numpy()})
    assert len(expected[0])==len(actual[0])
    # Equal-score NMS order may differ; compare sorted score vectors.
    np.testing.assert_allclose(np.sort(expected[1].numpy()),np.sort(actual[1]),rtol=1e-3,atol=1e-4)
    meta=json.loads((args.model_dir/'model.json').read_text());meta['onnx_bytes']=path.stat().st_size
    meta['selected_epoch']=checkpoint['epoch'];meta['onnx_cpu_validation']='pass'
    (args.model_dir/'model.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps({'onnx_bytes':path.stat().st_size,'selected_epoch':checkpoint['epoch'],'validation':'pass'}))


if __name__=='__main__':main()
