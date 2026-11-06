import numpy as np
import pytest

from service.multimodal_retrieval import score_vectors


def vec(cosine):
    return [cosine, np.sqrt(1 - cosine ** 2)]


def test_channel_calibration_corrects_cross_modality_cosine_bias():
    query = np.array([[1., 0.]])
    docs = [np.array([vec(.34), vec(.30)]), np.array([vec(.40)]),
            np.array([vec(.39), vec(.11)]), np.array([vec(.38), vec(.10)])]
    kinds = [['text', 'image_url'], ['text'], ['text', 'image_url'], ['text', 'image_url']]
    late = score_vectors(query, docs, kinds, ['text'])
    calibrated = score_vectors(query, docs, kinds, ['text'], mode='modality_zmax')
    assert late.argmax() == 1
    assert calibrated.argmax() == 0
    assert np.isfinite(calibrated).all()


@pytest.mark.parametrize('mode', ['late', 'modality_zmax', 'modality_rrf'])
def test_image_query_preserves_original_late_interaction(mode):
    query = np.array([[1., 0.]])
    docs = [np.array([[0., 1.], [1., 0.]]), np.array([vec(.8)])]
    kinds = [['text', 'image_url'], ['image_url']]
    np.testing.assert_equal(score_vectors(query, docs, kinds, ['image_url'], mode=mode), [1., .8])


@pytest.mark.parametrize('kind', ['text', 'image_url'])
@pytest.mark.parametrize('mode', ['modality_zmax', 'modality_rrf'])
def test_single_channel_falls_back_exactly(kind, mode):
    query = np.array([[1., 0.]])
    docs = [np.array([vec(.3)]), np.array([vec(.4)])]
    np.testing.assert_equal(score_vectors(query, docs, [[kind], [kind]], ['text'], mode=mode), [.3, .4])


@pytest.mark.parametrize('mode', ['modality_zmax', 'modality_rrf'])
def test_missing_and_constant_channels_have_no_nans(mode):
    query = np.array([[1., 0.]])
    docs = [np.array([vec(.4)]), np.array([vec(.3)]), np.array([vec(.4), vec(.3)])]
    scores = score_vectors(query, docs, [['text'], ['image_url'], ['text', 'image_url']], ['text'], mode=mode)
    assert np.isfinite(scores).all()
    if mode == 'modality_rrf':
        assert scores[2] > scores[0] and scores[2] > scores[1]
    else:
        np.testing.assert_equal(scores, [0, 0, 0])


def test_empty_corpus_and_invalid_shapes():
    query = np.array([[1., 0.]])
    assert score_vectors(query, [], [], ['text'], mode='modality_zmax').size == 0
    with pytest.raises(ValueError):
        score_vectors(query, [query], [[]], ['text'])
    with pytest.raises(ValueError):
        score_vectors(query, [np.array([[np.nan, 1.]])], [['text']], ['text'])


def test_context_propagation_is_one_hop_and_default_exact():
    from service.multimodal_retrieval import propagate_context
    raw = np.array([0., .2, 1.])
    args = (['u'] * 3, ['s'] * 3, [1, 2, 3])
    np.testing.assert_equal(propagate_context(raw, *args), raw)
    np.testing.assert_allclose(propagate_context(raw, *args, weight=.25), [.05, .4, 1])
    np.testing.assert_equal(raw, [0, .2, 1])


def test_context_boundaries_user_session_sequence_and_missing_vector():
    from service.multimodal_retrieval import propagate_context
    # Every adjacent pair has one independent boundary.
    raw = [0., 1., 0., 1., 0., 1., 0.]
    users = ['u', 'v', 'v', 'v', 'v', 'v', 'v']
    sessions = ['a', 'a', 'b', 'b', '', '', '']
    seq = [1, 2, 3, 5, 6, 7, 8]
    np.testing.assert_equal(propagate_context(raw, users, sessions, seq, .25), raw)
    np.testing.assert_equal(propagate_context([1, 0, 0], ['u']*3, ['s']*3, [1, 2, 3], .25,
                                            valid_vectors=[True, False, True]), [1, 0, 0])
    # A session reappearing after another session does not connect past it.
    np.testing.assert_equal(propagate_context([1, 0, 0], ['u']*3, ['a','b','a'], [1,2,3], .25), [1,0,0])
