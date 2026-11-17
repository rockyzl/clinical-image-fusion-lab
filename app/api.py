"""Local research deployment: restricted public cases, model version, audit, review."""
import json
import os
import sqlite3
import time
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Literal
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI,HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel,ConfigDict
from PIL import Image

ROOT=Path(__file__).resolve().parents[1];STATIC=ROOT/'app/static'
DB=Path(os.environ.get('FUSION_DEMO_AUDIT_DB',str(ROOT/'.runtime/inference.sqlite')))
app=FastAPI(title='Clinical image fusion research API',version='1.2')


@lru_cache
def manifest():
    path=STATIC/'assets/manifest.json'
    if not path.exists():raise HTTPException(503,'No verified model release is installed')
    return json.loads(path.read_text())


def database():
    DB.parent.mkdir(parents=True,exist_ok=True);db=sqlite3.connect(DB)
    db.execute('CREATE TABLE IF NOT EXISTS predictions (id TEXT PRIMARY KEY, case_id TEXT NOT NULL, version TEXT NOT NULL, payload TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, reviewed_at TEXT)')
    return db


@lru_cache
def session(name):
    release=manifest();info=release['encoder'] if name=='encoder' else release['models'].get(name)
    if not info:raise HTTPException(422,'Unknown model')
    path=STATIC/'assets'/info['file']
    import hashlib
    with path.open('rb') as f: actual=hashlib.file_digest(f,'sha256').hexdigest()
    if actual!=info['sha256']:raise HTTPException(503,'Model checksum failed')
    options=ort.SessionOptions();options.intra_op_num_threads=2
    return ort.InferenceSession(str(path),options,providers=['CPUExecutionProvider'])


class Inquiry(BaseModel):
    model_config=ConfigDict(extra='forbid')
    case_id:str
    model:Literal['clinical_linear','image_linear','concat_linear','metablock','healnet']='concat_linear'


class Review(BaseModel):
    model_config=ConfigDict(extra='forbid')
    action:Literal['accept','reject']
    demo_reviewer:Literal['research-reviewer']


@app.get('/health')
def health():
    release=manifest()
    return {'status':'ready','version':release['version'],'scope':'public research cases only','clinical_validation':False}


@app.get('/v1/model')
def model_info():
    m=manifest();return {k:m[k] for k in ['version','models','encoder','selection_sha256','seed','clinical_validation']}


@app.post('/v1/predict',status_code=201)
def predict(request:Inquiry):
    release=manifest();case=next((c for c in release['cases'] if c['id']==request.case_id),None)
    if case is None:raise HTTPException(404,'Unknown public case; arbitrary patient uploads are not supported')
    begin=time.perf_counter()
    image=np.array(Image.open(STATIC/case['image']).convert('RGB'),dtype=np.float32)/255
    if image.shape!=(224,224,3):raise HTTPException(503,'Unexpected released image shape')
    pixels=((image-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)).transpose(2,0,1)[None]
    features=session('encoder').run(None,{'pixels':pixels})[0]
    head=session(request.model);inputs={'image_features':features,'clinical':np.array([case['clinical']],np.float32)}
    logits=head.run(None,{i.name:inputs[i.name] for i in head.get_inputs()})[0][0]
    scores=np.exp(logits-logits.max());scores/=scores.sum()
    if not np.isfinite(scores).all():raise HTTPException(503,'Nonfinite model output')
    result={'id':str(uuid.uuid4()),'case_id':case['id'],'model':request.model,'version':release['version'],'seed':17,'scores':dict(zip(release['classes'],scores.astype(float).tolist())),'status':'pending_demo_review','latency_ms':round((time.perf_counter()-begin)*1000,2),'clinical_validation':False}
    with database() as db:db.execute('INSERT INTO predictions(id,case_id,version,payload,status) VALUES(?,?,?,?,?)',(result['id'],case['id'],release['version'],json.dumps(result),result['status']))
    return result


@app.post('/v1/research-review/{prediction_id}')
def review(prediction_id:str,request:Review):
    with database() as db:
        row=db.execute('SELECT status FROM predictions WHERE id=?',(prediction_id,)).fetchone()
        if row is None:raise HTTPException(404,'Prediction not found')
        if row[0]!='pending_demo_review':raise HTTPException(409,'This prediction was already reviewed')
        status='demo_accepted' if request.action=='accept' else 'demo_rejected'
        changed=db.execute('UPDATE predictions SET status=?,reviewed_at=CURRENT_TIMESTAMP WHERE id=? AND status=?',(status,prediction_id,'pending_demo_review'))
        if changed.rowcount != 1:raise HTTPException(409,'This prediction was already reviewed')
    return {'id':prediction_id,'status':status,'reviewer':request.demo_reviewer,'scope':'local demonstration; no clinical reviewer authentication'}


@app.get('/v1/predictions/{prediction_id}')
def prediction_record(prediction_id:str):
    with database() as db:
        row=db.execute('SELECT payload,status,created_at,reviewed_at FROM predictions WHERE id=?',(prediction_id,)).fetchone()
    if not row:raise HTTPException(404,'Prediction not found')
    result=json.loads(row[0]);result.update(status=row[1],created_at=row[2],reviewed_at=row[3]);return result


app.mount('/',StaticFiles(directory=STATIC,html=True),name='demo')
