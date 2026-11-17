"""Export a fixed seed, real case inputs, parity evidence and browser inference models."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
import pandas as pd
import torch
import onnxruntime as ort
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.transforms import functional as TF
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.common import ROOT, config, sha, write_json
from src.models import FusionModel, RawFeatureWrapper


class ImageEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Sequential(*list(resnet18(weights=None).children())[:-2])
        self.backbone.load_state_dict(torch.load(ROOT/'data/prepared/encoder.pt',weights_only=True))
    def forward(self, pixels):
        return self.backbone(pixels).permute(0,2,3,1).to(torch.float16).to(torch.float32)


def export(selection_path, test_path):
    c=config();torch.set_num_threads(4)
    selected=json.loads(Path(selection_path).read_text());test=json.loads(Path(test_path).read_text())
    if selected['mode']!='train' or test['split']!='test' or sha(selection_path)!=test['selection_sha256']:
        raise ValueError('Require completed, matching frozen training/test artifacts')
    dest=ROOT/'app/static/assets';dest.mkdir(exist_ok=True)
    cases_dir=dest/'cases';cases_dir.mkdir(exist_ok=True)
    rows=pd.read_csv(ROOT/'data/prepared/features.csv',dtype=str)
    # Fixed choice declared before test inspection: first two image IDs in each class;
    # no filtering on correctness. Browser demonstration uses seed17, not a test winner.
    predictions=np.load(Path(test_path).parent/'predictions.npz')
    case_indices=[]
    for label in range(6):
        local=np.flatnonzero(predictions['labels']==label)[:2]
        case_indices.extend(predictions['ids'][local].tolist())
    clinical=np.load(ROOT/'data/prepared/clinical.npy')
    cached=np.load(ROOT/'data/prepared/image_features.npy',mmap_mode='r')
    scaler=json.loads((ROOT/'data/prepared/image-scaler.json').read_text())
    encoder=ImageEncoder().eval(); models={}; exported={}
    for name in c['models']:
        trial=next(t for t in selected['trials'] if t['model']==name and t['seed']==17)
        if sha(trial['checkpoint'])!=trial['checkpoint_sha256']:raise ValueError('checkpoint drift')
        model=FusionModel(name,clinical.shape[1],c);model.load_state_dict(torch.load(trial['checkpoint'],weights_only=True)['weights'])
        wrapped=RawFeatureWrapper(model.eval(),scaler['mean'],scaler['scale']).eval();models[name]=wrapped
        path=dest/(name+'.onnx')
        torch.onnx.export(wrapped,(torch.zeros(1,7,7,512),torch.zeros(1,clinical.shape[1])),str(path),input_names=['image_features','clinical'],output_names=['logits'],opset_version=18,dynamo=True,external_data=False)
        exported[name]={'file':path.name,'sha256':sha(path),'checkpoint_sha256':trial['checkpoint_sha256'],'seed':17}
    encoder_path=dest/'image-encoder.onnx'
    torch.onnx.export(encoder,(torch.zeros(1,3,224,224),),str(encoder_path),input_names=['pixels'],output_names=['image_features'],opset_version=18,dynamo=True,external_data=False)
    options=ort.SessionOptions();options.intra_op_num_threads=2
    sessions={name:ort.InferenceSession(str(dest/m['file']),options,providers=['CPUExecutionProvider']) for name,m in exported.items()}
    image_session=ort.InferenceSession(str(encoder_path),options,providers=['CPUExecutionProvider'])
    records=[];parity=[]
    for index in case_indices:
        row=rows.iloc[index];pixels_source=Image.open(ROOT/'data/images'/row.img_id).convert('RGB')
        cropped=TF.center_crop(TF.resize(pixels_source,256,interpolation=TF.InterpolationMode.BILINEAR,antialias=True),[224,224])
        image_path=cases_dir/(row.img_id);cropped.info.clear();cropped.save(image_path)
        pixels=ResNet18_Weights.IMAGENET1K_V1.transforms()(pixels_source).unsqueeze(0)
        encoded_browser=image_session.run(None,{'pixels':pixels.numpy()})[0]
        with torch.inference_mode(): encoded_torch=encoder(pixels).numpy()
        np.testing.assert_allclose(encoded_browser,encoded_torch,atol=.016,rtol=.003)
        # Source cache was computed in batches; check the downstream probability effect.
        tab=clinical[index:index+1];image_cached=np.asarray(cached[index:index+1],dtype=np.float32)
        scores={}; max_errors={}
        for name,model in models.items():
            inputs={'image_features':encoded_browser,'clinical':tab}
            logits=sessions[name].run(None,{i.name:inputs[i.name] for i in sessions[name].get_inputs()})[0]
            prob=torch.tensor(logits).softmax(1).numpy()[0]
            with torch.inference_mode(): expected=model(torch.tensor(image_cached),torch.tensor(tab)).softmax(1).numpy()[0]
            np.testing.assert_allclose(prob,expected,atol=.003,rtol=.01)
            scores[name]=prob.tolist();max_errors[name]=float(np.max(np.abs(prob-expected)))
        local=int(np.flatnonzero(predictions['ids']==index)[0]); label=int(predictions['labels'][local])
        fields={k:(None if pd.isna(row[k]) else str(row[k])) for k in c['numeric_features']+c['categorical_features']}
        records.append({'id':row.img_id,'image':'assets/cases/'+row.img_id,'patient_id':row.patient_id,'lesion_id':row.lesion_id,'split':row.split,'fields':fields,'clinical':tab[0].tolist(),'published_label':c['classes'][label]})
        parity.append({'id':row.img_id,'max_probability_errors':max_errors,'expected_probabilities':scores})
    fig,axes=plt.subplots(5,3,figsize=(13,14),sharex=True)
    for axis,trial in zip(axes.flat,selected['trials']):
        curves=trial['curves'];epochs=[v['epoch'] for v in curves]
        axis.plot(epochs,[v['train_loss'] for v in curves],label='train',color='#2458a6')
        axis.plot(epochs,[v['tune_loss'] for v in curves],label='inner tune',color='#08796f')
        axis.axvline(trial['best_epoch'],color='#b08712',ls=':',label='selected epoch')
        axis.set_title(f"{trial['model']} · seed {trial['seed']}",fontsize=10);axis.grid(alpha=.15)
    axes[0,0].legend(fontsize=8);fig.supxlabel('Epoch');fig.supylabel('Weighted cross entropy');fig.tight_layout();fig.savefig(dest/'training-curves.svg');plt.close(fig)
    manifest={'version':'pad-fusion-v1.2','dataset':'PAD-UFES-20 v1','source_url':'https://data.mendeley.com/datasets/zr7vgbcyr2/1','license':'CC BY 4.0','classes':c['classes'],'seed':17,'cases':records,'models':exported,'encoder':{'file':encoder_path.name,'sha256':sha(encoder_path),'bytes':encoder_path.stat().st_size,'frozen':True},'selection_sha256':sha(selection_path),'test':test,'counts':json.loads((ROOT/'data/prepared/counts.json').read_text()),'source_repo':'','scientific_result':False,'clinical_validation':False,'selection_policy':'First two test image IDs in each class, regardless of model correctness. Seed17 chosen before export; no test-selected model.'}
    write_json(dest/'manifest.json',manifest)
    write_json(ROOT/'docs/export-parity.json',{'cases':parity,'maximum_allowed_absolute_probability_error':.003,'models':exported,'encoder_sha256':sha(encoder_path),'selected_seed':17,'test_sha256':sha(test_path)})
    print(json.dumps({'exported_models':len(models),'cases':len(records),'encoder_mb':encoder_path.stat().st_size/1e6,'parity':'passed'}))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--selection',required=True);p.add_argument('--test',required=True);a=p.parse_args();export(a.selection,a.test)
