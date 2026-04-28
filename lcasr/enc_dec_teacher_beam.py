import torch
from torch.nn import functional as F


@torch.no_grad()
def generate_enc_dec_beam(
        model,
        audio_signal=None,
        encoder_out=None,
        max_generate=256,
        bos_id=0,
        eos_id=0,
        beam_width=4,
        length_penalty=0.0,
        length_bonus=0.0,
        eos_min_length=0,
        eos_logit_margin=None,
        temperature=1.0,
    ):
    """Beam search for encoder-decoder teacher pseudo-transcripts.

    Returns a 1-D token tensor without BOS/EOS, matching the text_sequence
    convention expected by the existing teacher path.

    Note: temperature follows the repository's existing enc-dec convention in
    generate_enc_dec, where logits are multiplied by temperature. Values above
    1.0 sharpen the distribution; values below 1.0 soften it.
    """
    if encoder_out is None:
        assert audio_signal is not None, 'Either audio_signal or encoder_out must be provided'
        encoder_out = model.forward(audio_signal=audio_signal)

    a_hidden, length = encoder_out['a_hidden'], encoder_out['length']
    assert a_hidden.shape[0] == 1, 'generate_enc_dec_beam currently expects batch size 1'

    device = a_hidden.device
    beam_width = max(1, int(beam_width))
    temperature = max(float(temperature), 1e-6)

    def rank_score(score, tokens):
        generated_len = max(1, tokens.shape[-1] - 1)
        if length_penalty != 0.0:
            score = score / (generated_len ** length_penalty)
        if length_bonus != 0.0:
            score = score + (length_bonus * generated_len)
        return score

    active_beams = [(torch.tensor([[bos_id]], device=device, dtype=torch.long), 0.0)]
    finished_beams = []

    for _ in range(max_generate):
        candidates = []

        for tokens, score in active_beams:
            decoder_logits = model.language_model_decoder(
                tokens=tokens,
                a_hidden=a_hidden,
                a_lengths=length,
            )["logits"]

            next_log_probs = F.log_softmax(decoder_logits[0, -1, :] * temperature, dim=-1)

            generated_len = tokens.shape[-1] - 1
            if generated_len < eos_min_length:
                next_log_probs = next_log_probs.clone()
                next_log_probs[eos_id] = -float('inf')

            if eos_logit_margin is not None:
                next_log_probs = next_log_probs.clone()
                non_eos_log_probs = next_log_probs.clone()
                non_eos_log_probs[eos_id] = -float('inf')
                best_non_eos_log_prob = non_eos_log_probs.max()
                if next_log_probs[eos_id] < best_non_eos_log_prob + eos_logit_margin:
                    next_log_probs[eos_id] = -float('inf')

            top_log_probs, top_tokens = torch.topk(
                next_log_probs,
                k=min(beam_width, next_log_probs.shape[-1]),
            )

            for next_log_prob, next_token in zip(top_log_probs, top_tokens):
                next_token = int(next_token.item())
                next_score = score + float(next_log_prob.item())

                if next_token == eos_id:
                    finished_beams.append((tokens, next_score))
                else:
                    next_token_tensor = torch.tensor([[next_token]], device=device, dtype=torch.long)
                    next_tokens = torch.cat([tokens, next_token_tensor], dim=-1)
                    candidates.append((next_tokens, next_score))

        if len(candidates) == 0:
            break

        active_beams = sorted(
            candidates,
            key=lambda item: rank_score(item[1], item[0]),
            reverse=True,
        )[:beam_width]

    if len(finished_beams) == 0:
        finished_beams = active_beams

    best_tokens, _ = max(
        finished_beams,
        key=lambda item: rank_score(item[1], item[0]),
    )

    return best_tokens[:, 1:].squeeze(0).contiguous()


def patch_teacher_beam_generate(model, args):
    """Patch model.generate so the existing teacher path can use beam search.

    The patch only intercepts deterministic teacher calls that pass
    encoder_states, matching the pseudo-label path inside enc_dec_dynamic_eval.
    Stochastic calls (e.g. decode-agreement sampling) and ordinary
    model.generate calls are passed through to the original generate method.
    """
    if getattr(args, 'teacher_decode', 'greedy') != 'beam':
        return None

    original_generate = model.generate

    def beam_generate(audio_signal, *generate_args, **generate_kwargs):
        encoder_states = generate_kwargs.get('encoder_states', None)
        if generate_kwargs.get('sample', False) or encoder_states is None:
            return original_generate(audio_signal, *generate_args, **generate_kwargs)

        eos_logit_margin = getattr(args, 'teacher_eos_logit_margin', None)
        if eos_logit_margin is not None:
            eos_logit_margin = float(eos_logit_margin)

        text_sequence = generate_enc_dec_beam(
            model=model,
            audio_signal=audio_signal,
            encoder_out=encoder_states,
            max_generate=getattr(args, 'teacher_max_generate', 256),
            beam_width=getattr(args, 'teacher_beam_width', 4),
            length_penalty=getattr(args, 'teacher_length_penalty', 0.0),
            length_bonus=getattr(args, 'teacher_length_bonus', 0.0),
            eos_min_length=getattr(args, 'teacher_eos_min_length', 0),
            eos_logit_margin=eos_logit_margin,
            temperature=getattr(args, 'teacher_beam_temperature', 1.0),
        )
        return {"text_sequence": text_sequence.detach().cpu().tolist()}

    model.generate = beam_generate
    return original_generate
