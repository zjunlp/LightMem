"""Synthetic image-source metadata contracts; no external data or models."""
import base64
import io
import json

import httpx
import numpy as np
import pytest
from PIL import Image, TiffImagePlugin

import service.multimodal as mm
from service.multimodal_metadata import CaptureMetadata


def gazetteer(tmp_path, name='North City'):
    path = tmp_path / 'cities.txt'
    path.write_text(f'20\tLater City\tLater City\t\t1\t2\tP\tPPL\tXY\n'
                    f'10\t{name}\t{name}\t\t1\t2\tP\tPPL\tXY\n'
                    '30\tSouth City\tSouth City\t\t-1\t-2\tP\tPPL\tYZ\n')
    return path


def source_part(date='2024:03:04 12:34:56', gps=True, south=False):
    exif = Image.Exif()
    if date is not None:
        exif[34665] = {36867: date}
    if gps:
        r = TiffImagePlugin.IFDRational
        exif[34853] = {1: 'S' if south else 'N', 2: (r(1), r(0), r(0)),
                       3: 'W' if south else 'E', 4: (r(2), r(0), r(0))}
    data = io.BytesIO()
    Image.new('RGB', (16, 16), 'blue').save(data, format='JPEG', exif=exif)
    return dict(type='image_url', image_url=dict(
        url='data:image/jpeg;base64,' + base64.b64encode(data.getvalue()).decode()))


def test_capture_metadata_from_original_bytes_and_stable_geo_ties(tmp_path):
    parser = CaptureMetadata(gazetteer(tmp_path))
    part = source_part()
    original = [dict(type='text', text='untouched'), part]
    before = json.dumps(original, sort_keys=True)
    prepared = parser.augment(original)
    assert json.dumps(original, sort_keys=True) == before
    assert prepared[:2] == original and len(prepared) == 3
    assert prepared[-1]['text'] == ('Photo metadata. Capture date: 2024-03-04. '
        'Nearest GeoNames settlement to GPS: North City, country code XY; distance 0.0 km. '
        'This is a nearby gazetteer settlement, not a verified capture place.')
    assert len(parser.augment([part, part])) == 3
    assert 'South City' in parser.augment([source_part(south=True)])[-1]['text']


@pytest.mark.parametrize('offset', ['+08:00', '-05:30', '+00:00', '-00:00', None, 'invalid', '+25:00'])
def test_event_metadata_requires_original_valid_offset(offset):
    part = source_part()
    raw = base64.b64decode(part['image_url']['url'].split(',', 1)[1])
    with Image.open(io.BytesIO(raw)) as image:
        exif = image.getexif()
        details = exif.get_ifd(34665)
        if offset is not None:
            details[36881] = offset
        exif[34665] = details
        data = io.BytesIO()
        image.save(data, format='JPEG', exif=exif)
    result = CaptureMetadata.event_from_bytes(data.getvalue())
    if offset in ('+08:00', '-05:30', '+00:00'):
        assert result == dict(capture_datetime_iso='2024-03-04T12:34:56' + offset,
                              gps_latitude=1.0, gps_longitude=2.0)
    else:
        assert result is None


@pytest.mark.parametrize('date,gps', [(None, True), ('invalid', True),
                                     ('2024:03:04 12:34:56', False)])
def test_missing_or_invalid_capture_metadata_is_exact_noop(tmp_path, date, gps):
    parser = CaptureMetadata(gazetteer(tmp_path))
    parts = [source_part(date=date, gps=gps)]
    assert parser.augment(parts) == parts
    assert parser.text_from_bytes(b'not an image') is None


def test_index_fingerprint_and_invalid_records(tmp_path):
    path = gazetteer(tmp_path)
    first = CaptureMetadata(path)
    path = gazetteer(tmp_path, 'New City')
    assert CaptureMetadata(path).signature != first.signature
    path.write_text('bad row')
    with pytest.raises(ValueError):
        CaptureMetadata(path)
    path.write_text('')
    with pytest.raises(ValueError):
        CaptureMetadata(path)


def test_metadata_tokens_preserve_message_offsets_and_originals(tmp_path):
    parser = CaptureMetadata(gazetteer(tmp_path))
    batches = []
    def handler(request):
        body = json.loads(request.content)
        batches.append(body['input'])
        return httpx.Response(200, json={'data': [
            {'index': i, 'embedding': [1, 0] if 'image' in p else [0, 1]}
            for i, p in enumerate(body['input'])]})
    cfg = dict(base_url='https://embedding.example/v1', model='model', api_key='synthetic',
               client=httpx.Client(transport=httpx.MockTransport(handler)))
    service = mm.MemoryService(str(tmp_path / 'db'), remote_config=cfg, lexical_weight=0,
                               capture_metadata=parser)
    messages = [dict(role='user', content=[source_part()]),
                dict(role='assistant', content=[dict(type='text', text='original reply')])]
    request = mm.AddRequest.model_validate(dict(user_id='u', session_id='s', request_id='r', messages=messages))
    service.add(request)
    assert len(batches[0]) == 3
    assert 'image' in batches[0][0]
    assert batches[0][1]['text'].startswith('Photo metadata.')
    assert batches[0][2] == {'text': 'original reply'}
    with service.connect() as db:
        rows = db.execute('SELECT content,vector,encoder FROM memories ORDER BY sequence').fetchall()
    vector, types = mm.load_content(rows[0][1], lambda: json.loads(rows[0][0]))
    np.testing.assert_array_equal(vector, [[1, 0], [0, 1]])
    assert types == ['image_url', 'text']
    assert json.loads(rows[0][0]) == messages[0]['content']
    vector, types = mm.load_content(rows[1][1], lambda: json.loads(rows[1][0]))
    np.testing.assert_array_equal(vector, [[0, 1]])
    assert types == ['text']
    result = service.search(mm.SearchRequest(user_id='u', query='query', top_k=2))
    assert all(not any(p['type']=='text' and p['text'].startswith('Photo metadata.')
                       for p in item['content']) for item in result['data'])
    service.close()


def test_metadata_never_augments_query_and_lazy_vectors_use_same_profile(tmp_path):
    parser = CaptureMetadata(gazetteer(tmp_path))
    old = mm.MemoryService(str(tmp_path / 'db'))
    old.add(mm.AddRequest.model_validate(dict(user_id='u', session_id='s', request_id='r',
        messages=[dict(role='user', content=[source_part()])])))
    old.close()
    batches = []
    def handler(request):
        b = json.loads(request.content);batches.append(b)
        return httpx.Response(200, json={'data': [
            {'index': i, 'embedding': [1, 0]} for i in range(len(b['input']))]})
    cfg = dict(base_url='https://embedding.example/v1',model='model',api_key='synthetic',
               client=httpx.Client(transport=httpx.MockTransport(handler)))
    service = mm.MemoryService(str(tmp_path / 'db'), remote_config=cfg, capture_metadata=parser)
    service.encode_query_content([source_part()])
    assert len(batches[-1]['input']) == 1
    service.search(mm.SearchRequest(user_id='u', query='query', top_k=1))
    assert len(batches[-1]['input']) == 2
    with service.connect() as db:
        blob, signature = db.execute('SELECT vector,encoder FROM memories').fetchone()
    assert signature == service.signature and signature != service.encoder.signature
    assert mm.load_content(blob, [])[1] == ['image_url','text']
    service.close()
