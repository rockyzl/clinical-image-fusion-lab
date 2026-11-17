"""Exercise the actual exported models and the local persisted review lifecycle."""
import os,sys,tempfile,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
with tempfile.TemporaryDirectory() as tmp:
    os.environ['FUSION_DEMO_AUDIT_DB']=str(Path(tmp)/'test.sqlite')
    from app.api import app,manifest
    with TestClient(app) as client:
        assert client.get('/health').status_code==200
        assert client.post('/v1/predict',json={'case_id':'unknown'}).status_code==404
        case=manifest()['cases'][0]
        assert client.post('/v1/predict',json={'case_id':case['id'],'model':'unknown'}).status_code==422
        assert client.post('/v1/predict',json={'case_id':case['id'],'diagnosis':'inject'}).status_code==422
        response=client.post('/v1/predict',json={'case_id':case['id'],'model':'concat_linear'});assert response.status_code==201,response.text
        result=response.json();assert result['status']=='pending_demo_review'
        reference=json.loads((Path(__file__).resolve().parents[1]/'docs/export-parity.json').read_text())
        expected=next(row['expected_probabilities']['concat_linear'] for row in reference['cases'] if row['id']==case['id'])
        error=max(abs(result['scores'][c]-expected[i]) for i,c in enumerate(manifest()['classes']));assert error<.003,error
        route='/v1/research-review/'+result['id']
        assert client.post(route,json={'action':'accept','demo_reviewer':'real-doctor'}).status_code==422
        assert client.post(route,json={'action':'accept','demo_reviewer':'research-reviewer'}).status_code==200
        assert client.post(route,json={'action':'reject','demo_reviewer':'research-reviewer'}).status_code==409
        saved=client.get('/v1/predictions/'+result['id']).json();assert saved['status']=='demo_accepted' and saved['reviewed_at']
        print(json.dumps({'api':'passed','inference_max_error':error,'review_lifecycle':'pending -> demo accepted; duplicate rejected','checks':9}))
