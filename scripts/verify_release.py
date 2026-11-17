"""No training or dependencies: verify published model hashes and safe asset paths."""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'app/static';assets=root/'assets'
m=json.loads((assets/'manifest.json').read_text())
for entry in [m['encoder'],*m['models'].values()]:
 p=(assets/entry['file']).resolve()
 if assets.resolve() not in p.parents:raise ValueError('Unsafe model path')
 with p.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
 assert digest==entry['sha256'],p
for case in m['cases']:
 p=(root/case['image']).resolve();assert root.resolve() in p.parents and p.is_file()
 assert len(case['clinical'])==60 and case['split']=='test'
assert m['clinical_validation'] is False
assert m['test']['split']=='test' and len(m['test']['per_seed'])==15
print(f'Release verified: {len(m["cases"])} public cases and {len(m["test"]["per_seed"])} evaluated fits.')
