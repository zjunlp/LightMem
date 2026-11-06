import pytest
from service.multimodal_retrieval import session_diverse_order

def test_future_fixed_image_does_not_duplicate_head_session():
    ids=[0,1,2,3]; scores=[.9,None,.8,.7]; sessions=['s','s','t','u']
    assert session_diverse_order(ids,scores,sessions)==[0,1,2,3]
    assert session_diverse_order(ids,scores,sessions,protected_head=10)==[2,1,3,0]

def test_empty_sessions_remain_independent_and_tail_is_preserved():
    ids=list(range(103));scores=[None]+list(range(99));sessions=['']*100
    result=session_diverse_order(ids,scores,sessions,protected_head=10)
    assert result[0]==0 and result[100:]==ids[100:] and set(result)==set(ids)
    assert result[1:100]==list(range(99,0,-1))

@pytest.mark.parametrize('head',[True,-1,101,1.5,None])
def test_invalid_head_rejected(head):
    with pytest.raises(ValueError):session_diverse_order([0],[.1],['s'],protected_head=head)

@pytest.mark.parametrize('mode,expected',[('session_diverse',['a','image','b','c']),('session_reserved',['b','image','c','a'])])
@pytest.mark.parametrize('fail',[False,True])
def test_http_mode_reserves_future_image_and_preserves_fallback(tmp_path,monkeypatch,mode,expected,fail):
    import base64,io
    import numpy as np
    from PIL import Image
    from fastapi.testclient import TestClient
    import service.multimodal as mm
    import service.multimodal_reranker as runtime
    class Scorer:
        timeout=30
        def score(self,query,documents,decoder):
            if fail:raise RuntimeError('busy')
            return [{'a':.9,'b':.8,'c':.7}[documents[0][0]['text']]]
    monkeypatch.setattr(runtime,'LocalReranker',lambda *a,**kw:Scorer())
    monkeypatch.setattr(mm,'bm25_scores',lambda docs,text:np.arange(len(docs),0,-1))
    monkeypatch.setenv('MULTIMODAL_RERANKER_MODEL','synthetic')
    monkeypatch.setenv('MULTIMODAL_RERANKER_MODE',mode)
    monkeypatch.setenv('MULTIMODAL_PAIR_DIVERSITY','0')
    stream=io.BytesIO();Image.new('RGB',(2,2),'blue').save(stream,format='PNG')
    image={'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode()}}
    contents=[[{'type':'text','text':'a'}],[{'type':'text','text':'image'},image],[{'type':'text','text':'b'}],[{'type':'text','text':'c'}]]
    headers={'Authorization':'Bearer synthetic'}
    with TestClient(mm.create_app(str(tmp_path/'db'),api_key='synthetic')) as client:
        for i,(session,content) in enumerate(zip(['s','s','t','u'],contents)):
            response=client.post('/v1/memories/add',headers=headers,json={'user_id':'u','request_id':str(i),'session_id':session,'messages':[{'role':'user','content':content}]})
            assert response.status_code==200
        response=client.post('/v1/memories/search',headers=headers,json={'user_id':'u','query':'q','top_k':4})
        assert response.status_code==200
        items=response.json()['data']
        assert [item['content'][0]['text'] for item in items]==(['a','image','b','c'] if fail else expected)
        assert next(item['content'] for item in items if item['content'][0]['text']=='image')==contents[1]
        health=client.get('/health').json()
        assert bool(health['reranker_error'])==fail
