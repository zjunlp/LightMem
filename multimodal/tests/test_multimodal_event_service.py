"""Event completion through the API with synthetic originals and encoder output."""
import base64
import io
import json

import httpx
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image, TiffImagePlugin

import service.multimodal as mm
from service.multimodal_metadata import CaptureMetadata


def image_part():
    r = TiffImagePlugin.IFDRational
    exif = Image.Exif()
    exif[34665] = {36867: '2024:01:01 12:00:00', 36881: '+00:00'}
    exif[34853] = {1:'N',2:(r(1),r(0),r(0)),3:'E',4:(r(2),r(0),r(0))}
    stream = io.BytesIO()
    Image.new('RGB',(8,8)).save(stream,format='JPEG',exif=exif)
    return dict(type='image_url',image_url=dict(url='data:image/jpeg;base64,'+base64.b64encode(stream.getvalue()).decode()))


def test_http_event_completion_cache_scope_and_disabled_parity(tmp_path, monkeypatch):
    city = tmp_path/'cities.txt'
    city.write_text('1\tCity\tCity\t\t1\t2\tP\tPPL\tXY\n')
    def handler(request):
        body = json.loads(request.content)
        return httpx.Response(200,json={'data':[dict(index=i,embedding=[1,0]) for i in range(len(body['input']))]})
    cfg = dict(base_url='https://synthetic.invalid/v1',model='synthetic',api_key='synthetic',client=httpx.Client(transport=httpx.MockTransport(handler)))
    service = mm.MemoryService(str(tmp_path/'db'),remote_config=cfg,lexical_weight=0,
                               capture_metadata=CaptureMetadata(city),event_completion=True)
    monkeypatch.setattr(mm,'score_vectors',lambda query,vectors,*args,**kwargs: np.arange(len(vectors),0,-1,dtype=float))
    monkeypatch.setattr(mm,'MemoryService',lambda *args,**kwargs: service)
    parse = CaptureMetadata.event_from_bytes
    parsed = []
    def counted(raw):
        parsed.append(1)
        return parse(raw)
    monkeypatch.setattr(CaptureMetadata,'event_from_bytes',staticmethod(counted))
    headers={'Authorization':'Bearer secret'}
    with TestClient(mm.create_app(str(tmp_path/'db'),api_key='secret')) as client:
        messages=[dict(role='user',content=[image_part() if i in (0,104) else dict(type='text',text=f'item {i}')]) for i in range(105)]
        for user in ('u','other'):
            added=client.post('/add',headers=headers,json=dict(user_id=user,request_id='r',session_id='s',messages=messages))
            assert added.status_code == 200,added.text
        with service.connect() as db:
            ids=[row[0] for row in db.execute('SELECT id FROM memories WHERE user_id=? ORDER BY sequence',('u',))]
        query=dict(user_id='u',query='query',top_k=100)
        result=client.post('/search',headers=headers,json=query)
        assert result.status_code == 200,result.text
        expected=ids[:20]+[ids[20],ids[104],ids[21],ids[22],ids[23]]+ids[24:99]
        assert [r['id'] for r in result.json()['data']] == expected
        assert len(parsed)==2
        assert client.post('/search',headers=headers,json=query).json() == result.json()
        assert len(parsed)==2
        with monkeypatch.context() as limited:
            limited.setattr(mm,'MAX_MEDIA_BYTES',mm.image_bytes(messages[0]['content'][0]))
            bounded=client.post('/search',headers=headers,json=query)
            assert bounded.status_code == 200
            # The inserted second image overflows: stop there, including text
            # already encountered, without skipping ahead to later evidence.
            assert [r['id'] for r in bounded.json()['data']] == ids[:21]
        with service.connect() as db:
            assert db.execute('SELECT COUNT(*) FROM capture_events WHERE user_id=?',('u',)).fetchone()[0]==105
            assert db.execute('SELECT COUNT(*) FROM capture_events WHERE user_id=?',('other',)).fetchone()[0]==0
        service.event_completion=False
        baseline=client.post('/search',headers=headers,json=query)
        assert [r['id'] for r in baseline.json()['data']] == ids[:100]
        assert len(parsed)==2
