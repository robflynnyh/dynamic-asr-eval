"""Verify the new batched ``model.generate`` matches the old reference impl.

Loads a real EncDecSconformerV2 checkpoint AND real TEDLium audio, then compares
the *new* batched ``model.generate`` against a verbatim copy of the *old*
single-batch impl baked in below as ``OLD_model_generate``.

Tests:
    - test_greedy_single        : new vs old    (B=1, greedy)
    - test_max_generate_cap     : new vs old token equality at cap; known prob length delta
    - test_sampling_single      : new vs old    (B=1, sample, same seed)
    - test_beam_search_runs     : autoregressive beam path returns a valid capped decode
    - test_greedy_rollouts      : num_rollouts=N greedy => all rows match B=1 greedy
    - test_greedy_true_batch    : true B=2 greedy => row-i matches old on row i
    - test_eos_early_stop       : forced immediate EOS returns []
    - test_beam_search_eos      : beam path preserves immediate-EOS finalization
    - test_eos_drop_in_batch    : one row EOS-drops mid-loop, surviving row matches ref

Override with CHECKPOINT=/path/to/step.pt and AUDIO=/path/to/file.sph env vars.

Run with::

    cd lcasr && python3.10 test_cases/test_generate_batched.py
"""
import os
import sys
import torch

# So that ``import lcasr`` resolves to the installed long-context-asr package
# while local sibling dirs (tedlium/, etc.) are also reachable.
_HERE = os.path.dirname(os.path.abspath(__file__))
_LCASR_DIR = os.path.dirname(_HERE)  # .../dynamic-asr-eval/lcasr
if _LCASR_DIR not in sys.path:
    sys.path.insert(0, _LCASR_DIR)

import lcasr
from lcasr.utils.general import load_model, get_model_class
from lcasr.utils.audio_tools import processing_chain

DEFAULT_CKPT = "/store/store5/data/acp21rjf_checkpoints/lcasr/enc_dec_v2/step_105360.pt"
DEFAULT_AUDIO = "/store/store4/data/TEDLIUM_release1/legacy/test/sph/AimeeMullins_2009P.sph"
DEFAULT_AUDIO_2 = "/store/store4/data/TEDLIUM_release1/legacy/test/sph/BillGates_2010.sph"


# ---------------------------------------------------------------------------
# Reference implementations (verbatim copies of the OLD code, pre-refactor)
# ---------------------------------------------------------------------------

@torch.no_grad()
def OLD_model_generate(
        model,
        audio_signal,
        max_generate='encoder_states',
        bos_id=0,
        eos_id=0,
        return_encoder_states=False,
        return_ctc_states=False,
        prompt=None,
        encoder_states=None,
        remove_prompt=True,
        sample=False,
        temperature=1.0,
    ):
    """Single-batch, kv-cached generate as it existed before the refactor."""
    if encoder_states is None:
        encoder_out = model.forward(audio_signal=audio_signal)
        a_hidden, length = encoder_out['a_hidden'], encoder_out['length']
    else:
        a_hidden, length = encoder_states['a_hidden'], encoder_states['length']

    if max_generate == 'encoder_states':
        max_generate = length.max().item()

    if prompt is None:
        text_sequence = torch.LongTensor([[bos_id]])
    elif isinstance(prompt, list):
        text_sequence = torch.LongTensor([prompt])
    else:
        text_sequence = prompt
    text_sequence = text_sequence.to(a_hidden.device)
    prompt_length = text_sequence.shape[1]

    finished = False
    cache = None
    if text_sequence.ndim == 3:
        prompt_length = 0
        final_text_sequence = torch.LongTensor([[]]).to(a_hidden.device)
    else:
        final_text_sequence = text_sequence.clone()
    if sample is False:
        temperature = 1.0
    steps = 0
    all_probs = []

    while not finished:
        decoder_out = model.language_model_decoder(
            tokens=text_sequence,
            a_hidden=a_hidden,
            a_lengths=length,
            cache=cache,
            text_lengths=torch.tensor([text_sequence.shape[1]]).to(a_hidden.device),
        )
        decoder_logits = decoder_out['logits']
        cache = decoder_out['kv_cache']

        logits = decoder_logits[0, -1, :]
        probs = (logits / temperature).softmax(dim=-1)
        if sample is False:
            decoder_pred = probs.argmax(dim=-1)
        else:
            decoder_pred = probs.multinomial(num_samples=1).squeeze(0)

        all_probs.append(probs[decoder_pred].item())
        steps += 1
        if decoder_pred == eos_id or (steps > max_generate):
            finished = True
        else:
            text_sequence = decoder_pred.unsqueeze(0).unsqueeze(0)
            final_text_sequence = torch.cat([final_text_sequence, text_sequence], dim=1)

    final_text_sequence = final_text_sequence.squeeze(0).cpu().tolist()
    if remove_prompt:
        final_text_sequence = final_text_sequence[prompt_length:]
    return {'text_sequence': final_text_sequence, 'probs': all_probs}


@torch.no_grad()
def OLD_generate_enc_dec(
        model,
        audio_signal,
        max_generate=256,
        bos_id=0,
        eos_id=0,
        sample=1,
        greedy=True,
        temperature=1.0,
    ):
    """Verbatim copy of the deleted ``lib.generate_enc_dec`` (uncached, batched).

    Used to verify that ``model.generate`` reproduces the same greedy output as
    the function it replaced at lib.py:1141.
    """
    encoder_out = model.forward(audio_signal=audio_signal)
    a_hidden, length = encoder_out['a_hidden'], encoder_out['length']
    a_hidden = a_hidden.repeat(sample, 1, 1)
    text_sequence = torch.LongTensor([[bos_id]]).to(a_hidden.device).repeat(sample, 1)
    finised_sequences = []
    finished = False
    while not finished:
        decoder_logits = model.language_model_decoder(
            tokens=text_sequence,
            a_hidden=a_hidden,
            a_lengths=length,
        )["logits"]

        probs = (decoder_logits[:, -1, :] * temperature).softmax(dim=-1)
        if sample == 1 and greedy:
            decoder_pred = probs.argmax(dim=-1)[None]
        else:
            decoder_pred = torch.multinomial(probs, num_samples=1)

        indices_to_drop = 0
        new_text_sequences = []
        for i in range(sample):
            if decoder_pred[i] == eos_id or text_sequence[i].shape[0] > max_generate:
                finised_sequences.append(text_sequence[i])
                indices_to_drop += 1
            else:
                new_text_sequences.append(torch.cat([text_sequence[i, None], decoder_pred[i, None]], dim=1))
        if indices_to_drop > 0:
            a_hidden = a_hidden[:-indices_to_drop]
        if len(new_text_sequences) > 0:
            text_sequence = torch.cat(new_text_sequences, dim=0)
        sample = a_hidden.shape[0]
        if sample == 0:
            finished = True

    text_lengths = torch.LongTensor([el.shape[0] for el in finised_sequences])
    text_sequence = torch.nn.utils.rnn.pad_sequence(finised_sequences, batch_first=True, padding_value=0)
    text_sequence = text_sequence[:, 1:]  # remove bos
    text_lengths -= 1
    return text_sequence, encoder_out, text_lengths


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

_MODEL = None
_TOKENIZER = None


def get_model():
    """Load the real checkpoint once and cache it."""
    global _MODEL, _TOKENIZER
    if _MODEL is not None:
        return _MODEL, _TOKENIZER

    ckpt_path = os.environ.get("CHECKPOINT", DEFAULT_CKPT)
    print(f"loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = checkpoint['config']
    # Disable flash so this runs on CPU and on cards without flash-attn.
    if hasattr(cfg, 'model'):
        cfg.model.flash_attn = False

    tokenizer = lcasr.utils.audio_tools.load_tokenizer()
    model = load_model(cfg, model_class=get_model_class(cfg), vocab_size=len(tokenizer))
    model.load_state_dict(checkpoint['model'], strict=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.device = device
    model = model.to(device).eval()
    print(f"model on {device}")

    _MODEL, _TOKENIZER = model, tokenizer
    return _MODEL, _TOKENIZER


_AUDIO_CACHE = {}


def load_audio_chunk(path, model, T=2048, offset=0):
    """Load TEDLium-style audio, run the spec processing chain, and slice
    a contiguous (1, feat_in, T) chunk starting at frame `offset`."""
    key = (path, T, offset)
    if key in _AUDIO_CACHE:
        return _AUDIO_CACHE[key]
    spec = processing_chain(path)  # shape (1, feat_in, T_full)
    if spec.shape[-1] < offset + T:
        raise RuntimeError(
            f"audio at {path} too short: have {spec.shape[-1]} frames, need {offset + T}"
        )
    chunk = spec[..., offset:offset + T].to(model.device)
    _AUDIO_CACHE[key] = chunk
    return chunk


def make_audio(model, T=2048, seed=0):
    """Pick a real audio file and slice a stable chunk for the test.

    `seed` selects which file/offset to use so we can build a multi-row batch
    from genuinely different speakers."""
    audio_path = os.environ.get("AUDIO", DEFAULT_AUDIO)
    audio_path_2 = os.environ.get("AUDIO_2", DEFAULT_AUDIO_2)
    # seed == 0 -> first file head; seed == 1 -> second file head.
    if seed == 0:
        return load_audio_chunk(audio_path, model, T=T, offset=0)
    elif seed == 1:
        return load_audio_chunk(audio_path_2, model, T=T, offset=0)
    else:
        return load_audio_chunk(audio_path, model, T=T, offset=seed * T)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

MAX_GEN = 16


def _decode(tokenizer, ids):
    return tokenizer.decode([i for i in ids if i >= 0]).strip()


def assert_probs_close(new_probs, old_probs, context):
    assert len(new_probs) == len(old_probs), (
        f"{context}: prob length mismatch: new={len(new_probs)} old={len(old_probs)}"
    )
    for i, (a, b) in enumerate(zip(new_probs, old_probs)):
        assert abs(a - b) < 1e-5, f"{context}: prob diverged at step {i}: {a} vs {b}"


def test_greedy_single():
    m, tok = get_model()
    x = make_audio(m)

    new = m.generate(x, max_generate=MAX_GEN)
    old = OLD_model_generate(m, x, max_generate=MAX_GEN)

    new_seq = new['text_sequence'][0]
    new_probs = new['probs'][0]
    old_probs = old['probs']

    print(f"  new tokens: {new_seq}")
    print(f"  old tokens: {old['text_sequence']}")
    print(f"  decoded:    {_decode(tok, new_seq)!r}")

    assert new_seq == old['text_sequence'], (
        f"greedy mismatch:\n new={new_seq}\n old={old['text_sequence']}"
    )

    assert_probs_close(new_probs, old_probs, "greedy")
    print("PASS test_greedy_single")


def test_max_generate_cap():
    """With EOS disabled, both impls emit the same capped tokens.

    The old cached implementation performs one extra decoder step after it
    has already emitted max_generate tokens, so it has one extra probability
    for an un-emitted cap-step token. Keep that difference explicit instead
    of hiding it in the normal greedy test.
    """
    m, tok = get_model()
    x = make_audio(m)

    new = m.generate(x, max_generate=MAX_GEN, eos_id=-1)
    old = OLD_model_generate(m, x, max_generate=MAX_GEN, eos_id=-1)

    new_seq = new['text_sequence'][0]
    old_seq = old['text_sequence']
    new_probs = new['probs'][0]
    old_probs = old['probs']

    print(f"  capped decoded: {_decode(tok, new_seq)!r}")
    assert len(new_seq) == MAX_GEN, f"new did not reach cap: len={len(new_seq)}"
    assert new_seq == old_seq, f"cap tokens mismatch:\n new={new_seq}\n old={old_seq}"
    assert len(new_probs) == MAX_GEN, f"new probs should match emitted cap: {len(new_probs)}"
    assert len(old_probs) == MAX_GEN + 1, (
        f"old should include one un-emitted cap-step prob, got {len(old_probs)}"
    )
    for i, (a, b) in enumerate(zip(new_probs, old_probs[:MAX_GEN])):
        assert abs(a - b) < 1e-5, f"cap prob diverged at step {i}: {a} vs {b}"
    print("PASS test_max_generate_cap")


def test_sampling_single():
    m, tok = get_model()
    x = make_audio(m)

    torch.manual_seed(7)
    new = m.generate(x, max_generate=MAX_GEN, sample=True, temperature=1.0)
    torch.manual_seed(7)
    old = OLD_model_generate(m, x, max_generate=MAX_GEN, sample=True, temperature=1.0)

    print(f"  new tokens: {new['text_sequence'][0]}")
    print(f"  old tokens: {old['text_sequence']}")
    print(f"  decoded:    {_decode(tok, new['text_sequence'][0])!r}")
    assert new['text_sequence'][0] == old['text_sequence'], (
        f"sampling mismatch (same seed):\n new={new['text_sequence'][0]}\n old={old['text_sequence']}"
    )
    assert_probs_close(new['probs'][0], old['probs'], "sampling")
    print("PASS test_sampling_single")


def test_beam_search_runs():
    m, tok = get_model()
    x = make_audio(m)

    out = m.generate(x, max_generate=MAX_GEN, beam_width=2, return_beam_scores=True)
    seq = out['text_sequence'][0]
    probs = out['probs'][0]

    print(f"  beam decoded: {_decode(tok, seq)!r}")
    assert isinstance(seq, list), f"beam seq should be a list, got {type(seq)}"
    assert len(seq) <= MAX_GEN, f"beam output exceeded cap: {len(seq)} > {MAX_GEN}"
    assert len(probs) <= MAX_GEN, f"beam probs exceeded cap: {len(probs)} > {MAX_GEN}"
    assert len(out['beam_scores']) == 1
    print("PASS test_beam_search_runs")


def test_greedy_rollouts():
    """num_rollouts=N greedy: every row must equal the single-row greedy AND
    that single-row greedy must equal OLD_model_generate (so we don't just
    verify self-consistent broadcasting on garbage)."""
    m, tok = get_model()
    x = make_audio(m)

    single = m.generate(x, max_generate=MAX_GEN)['text_sequence'][0]
    ref = OLD_model_generate(m, x, max_generate=MAX_GEN)['text_sequence']
    assert single == ref, f"single-row greedy drift vs old: {single} vs {ref}"

    multi = m.generate(x, max_generate=MAX_GEN, num_rollouts=4)['text_sequence']
    print(f"  single greedy: {_decode(tok, single)!r}")
    assert len(multi) == 4
    for i, row in enumerate(multi):
        assert row == single, f"rollout {i} drifted:\n {row}\n vs {single}"
    print("PASS test_greedy_rollouts")


def test_greedy_true_batch():
    m, tok = get_model()
    x0 = make_audio(m, seed=0)
    x1 = make_audio(m, seed=1)
    x_batch = torch.cat([x0, x1], dim=0)

    batched = m.generate(x_batch, max_generate=MAX_GEN)['text_sequence']
    assert len(batched) == 2

    ref0 = OLD_model_generate(m, x0, max_generate=MAX_GEN)['text_sequence']
    ref1 = OLD_model_generate(m, x1, max_generate=MAX_GEN)['text_sequence']

    print(f"  row 0: {_decode(tok, batched[0])!r}")
    print(f"  row 1: {_decode(tok, batched[1])!r}")
    assert batched[0] == ref0, f"row 0: {batched[0]} vs {ref0}"
    assert batched[1] == ref1, f"row 1: {batched[1]} vs {ref1}"
    print("PASS test_greedy_true_batch")


def test_eos_early_stop():
    """Set eos_id to whatever greedy picks at step 1 -> immediate EOS, empty seq."""
    m, _ = get_model()
    x = make_audio(m)

    seq = m.generate(x, max_generate=MAX_GEN, eos_id=-1)['text_sequence'][0]
    first_token = seq[0]

    out = m.generate(x, max_generate=MAX_GEN, eos_id=first_token)
    ref = OLD_model_generate(m, x, max_generate=MAX_GEN, eos_id=first_token)
    assert out['text_sequence'][0] == [], (
        f"expected empty sequence (immediate EOS), got {out['text_sequence'][0]}"
    )
    assert out['text_sequence'][0] == ref['text_sequence'], (
        f"new immediate EOS drifted from old: {out['text_sequence'][0]} vs {ref['text_sequence']}"
    )
    assert_probs_close(out['probs'][0], ref['probs'], "immediate eos")
    print("PASS test_eos_early_stop")


def test_beam_search_eos_early_stop():
    """Set eos_id to greedy step 1. Since EOS is the top first-step token, the
    finished empty beam should remain best under log-prob scoring."""
    m, _ = get_model()
    x = make_audio(m)

    seq = m.generate(x, max_generate=MAX_GEN, eos_id=-1)['text_sequence'][0]
    first_token = seq[0]

    out = m.generate(x, max_generate=MAX_GEN, eos_id=first_token, beam_width=2)
    assert out['text_sequence'][0] == [], (
        f"expected empty beam sequence (immediate EOS), got {out['text_sequence'][0]}"
    )
    print("PASS test_beam_search_eos_early_stop")


def test_eos_drop_in_batch():
    """Force row 0 to EOS at step 1 while row 1 keeps generating. Both rows must
    match the corresponding single-row OLD references — verifies the drop logic
    doesn't corrupt cache slicing for the surviving row, AND the dropped row
    finalizes correctly."""
    m, _ = get_model()
    x0 = make_audio(m, seed=0)
    x1 = make_audio(m, seed=1)
    x_batch = torch.cat([x0, x1], dim=0)

    row0_seq_uncapped = m.generate(x0, max_generate=MAX_GEN, eos_id=-1)['text_sequence'][0]
    early_eos = row0_seq_uncapped[0]

    ref0 = OLD_model_generate(m, x0, max_generate=MAX_GEN, eos_id=early_eos)['text_sequence']
    ref1 = OLD_model_generate(m, x1, max_generate=MAX_GEN, eos_id=early_eos)['text_sequence']
    assert ref0 == [], f"sanity: forced eos should yield empty ref0, got {ref0}"

    batched = m.generate(x_batch, max_generate=MAX_GEN, eos_id=early_eos)['text_sequence']
    assert len(batched) == 2
    assert batched[0] == ref0, f"row 0: {batched[0]} vs {ref0}"
    assert batched[1] == ref1, f"row 1 drifted post-drop: {batched[1]} vs {ref1}"
    print("PASS test_eos_drop_in_batch")


def test_generate_enc_dec_replacement_greedy():
    """Verify the lib.py:1141 replacement: new ``model.generate`` (greedy, B=1)
    produces the same tokens as the deleted ``generate_enc_dec`` (uncached)
    that it now stands in for. Cached vs uncached forward passes are
    algebraically equivalent so greedy argmax should agree."""
    m, tok = get_model()
    x = make_audio(m)

    new_seq = m.generate(x, max_generate=MAX_GEN)['text_sequence'][0]
    old_tensor, _, _ = OLD_generate_enc_dec(m, x, max_generate=MAX_GEN)
    old_seq = old_tensor[0].tolist()
    print(f"  new (len {len(new_seq)}): {new_seq}")
    print(f"  old (len {len(old_seq)}): {old_seq}")
    print(f"  decoded:    {_decode(tok, new_seq)!r}")
    assert new_seq == old_seq, (
        f"generate_enc_dec replacement diverged:\n new={new_seq}\n old={old_seq}"
    )
    print("PASS test_generate_enc_dec_replacement_greedy")


if __name__ == "__main__":
    test_greedy_single()
    test_max_generate_cap()
    test_sampling_single()
    test_beam_search_runs()
    test_greedy_rollouts()
    test_greedy_true_batch()
    test_eos_early_stop()
    test_beam_search_eos_early_stop()
    test_eos_drop_in_batch()
    test_generate_enc_dec_replacement_greedy()
    print("\nAll tests passed.")
