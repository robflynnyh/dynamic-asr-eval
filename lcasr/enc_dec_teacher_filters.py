from difflib import SequenceMatcher
import re


def add_enc_dec_teacher_filter_args(parser):
    parser.add_argument(
        '--teacher_filter_max_length',
        action='store_true',
        help='Skip teacher updates when the generated token count is implausibly large for the chunk length',
    )
    parser.add_argument(
        '--teacher_min_frames_per_token',
        type=int,
        default=8,
        help='Maximum generated teacher token count is spectrogram frames divided by this value',
    )
    parser.add_argument(
        '--teacher_filter_max_consecutive_token_repeat',
        action='store_true',
        help='Skip teacher updates when the same token repeats too many times consecutively',
    )
    parser.add_argument(
        '--teacher_max_consecutive_token_repeat',
        type=int,
        default=3,
        help='Maximum allowed consecutive repeats of the same teacher token',
    )
    parser.add_argument(
        '--teacher_filter_repeated_token_ngrams',
        action='store_true',
        help='Skip teacher updates when token n-grams loop consecutively',
    )
    parser.add_argument(
        '--teacher_repeated_token_ngram_sizes',
        type=int,
        nargs='+',
        default=[2, 3],
        help='Token n-gram sizes to check for consecutive loops',
    )
    parser.add_argument(
        '--teacher_repeated_token_ngram_min_repeats',
        type=int,
        default=2,
        help='Minimum number of consecutive repeats of a token n-gram before skipping',
    )
    parser.add_argument(
        '--teacher_filter_decode_agreement',
        action='store_true',
        help='Skip teacher updates when the teacher decode disagrees with a second sampled decode',
    )
    parser.add_argument(
        '--teacher_decode_agreement_temperature',
        type=float,
        default=0.7,
        help='Sampling temperature for the second decode used in the agreement filter',
    )
    parser.add_argument(
        '--teacher_decode_agreement_min_similarity',
        type=float,
        default=0.65,
        help='Minimum SequenceMatcher similarity for the second decode agreement filter',
    )
    parser.add_argument(
        '--teacher_filter_low_confidence',
        action='store_true',
        help='Skip teacher updates when the forced teacher path is too low-confidence',
    )
    parser.add_argument(
        '--teacher_min_mean_max_prob',
        type=float,
        default=0.35,
        help='Minimum mean max probability across teacher-forced decoder steps',
    )
    parser.add_argument(
        '--teacher_max_mean_entropy',
        type=float,
        default=2.5,
        help='Maximum mean entropy across teacher-forced decoder steps',
    )
    parser.add_argument(
        '--teacher_filter_repeated_words',
        action='store_true',
        help='Skip teacher updates when a decoded word repeats too many times consecutively',
    )
    parser.add_argument(
        '--teacher_max_consecutive_word_repeat',
        type=int,
        default=3,
        help='Maximum allowed consecutive repeats of the same decoded word',
    )
    parser.add_argument(
        '--teacher_filter_ctc_agreement',
        action='store_true',
        help='Skip teacher updates when the encoder-decoder teacher text disagrees with the CTC branch',
    )
    parser.add_argument(
        '--teacher_ctc_agreement_min_similarity',
        type=float,
        default=0.5,
        help='Minimum word-level SequenceMatcher similarity between encoder-decoder and CTC text',
    )
    return parser


def _sequence_similarity(first, second):
    return SequenceMatcher(a=list(first), b=list(second)).ratio()


def _word_sequence(text):
    return re.findall(r"[a-z0-9']+", text.lower())


def _longest_consecutive_repeat(sequence):
    longest_repeat = 0
    longest_item = None
    current_repeat = 0
    previous_item = None

    for item in sequence:
        if item == previous_item:
            current_repeat += 1
        else:
            previous_item = item
            current_repeat = 1

        if current_repeat > longest_repeat:
            longest_repeat = current_repeat
            longest_item = item

    return longest_repeat, longest_item


def _find_repeated_ngram_loop(sequence, ngram_size, min_repeats):
    total_ngram_span = ngram_size * min_repeats
    if ngram_size <= 0 or min_repeats <= 1 or len(sequence) < total_ngram_span:
        return False, (), 0

    for start in range(len(sequence) - total_ngram_span + 1):
        ngram = tuple(sequence[start:start + ngram_size])
        repeat_count = 1
        cursor = start + ngram_size

        while cursor + ngram_size <= len(sequence):
            if tuple(sequence[cursor:cursor + ngram_size]) != ngram:
                break
            repeat_count += 1
            cursor += ngram_size

        if repeat_count >= min_repeats:
            return True, ngram, repeat_count

    return False, (), 0


def should_skip_faulty_teacher_prediction(
        args,
        teacher_pred_tokens,
        teacher_pred_text,
        spec_frames,
        agreement_tokens=None,
        teacher_mean_max_prob=None,
        teacher_mean_entropy=None,
        ctc_text=None,
    ):
    if args.__dict__.get('teacher_filter_max_length', False):
        min_frames_per_token = args.__dict__.get('teacher_min_frames_per_token', 8)
        if min_frames_per_token > 0:
            max_teacher_tokens = spec_frames / min_frames_per_token
            if len(teacher_pred_tokens) > max_teacher_tokens:
                return True, (
                    f'too many teacher tokens ({len(teacher_pred_tokens)} tokens for {spec_frames} frames; '
                    f'max {max_teacher_tokens:.2f})'
                )

    if args.__dict__.get('teacher_filter_max_consecutive_token_repeat', False):
        longest_repeat, longest_repeat_token = _longest_consecutive_repeat(teacher_pred_tokens)
        max_consecutive_repeat = args.__dict__.get('teacher_max_consecutive_token_repeat', 3)
        if longest_repeat > max_consecutive_repeat:
            return True, (
                f'teacher token {longest_repeat_token} repeated {longest_repeat} times consecutively '
                f'(limit {max_consecutive_repeat})'
            )

    if args.__dict__.get('teacher_filter_repeated_token_ngrams', False):
        min_repeats = args.__dict__.get('teacher_repeated_token_ngram_min_repeats', 2)
        ngram_sizes = sorted(set(args.__dict__.get('teacher_repeated_token_ngram_sizes', [2, 3])))
        for ngram_size in ngram_sizes:
            repeated, ngram, repeat_count = _find_repeated_ngram_loop(
                teacher_pred_tokens,
                ngram_size,
                min_repeats,
            )
            if repeated:
                return True, (
                    f'teacher token {ngram_size}-gram {list(ngram)} repeated {repeat_count} times consecutively'
                )

    if args.__dict__.get('teacher_filter_decode_agreement', False) and agreement_tokens is not None:
        min_similarity = args.__dict__.get('teacher_decode_agreement_min_similarity', 0.65)
        similarity = _sequence_similarity(teacher_pred_tokens, agreement_tokens)
        if similarity < min_similarity:
            return True, (
                f'teacher decode agreement too low ({similarity:.2f} < {min_similarity:.2f})'
            )

    if args.__dict__.get('teacher_filter_low_confidence', False):
        min_mean_max_prob = args.__dict__.get('teacher_min_mean_max_prob', 0.35)
        max_mean_entropy = args.__dict__.get('teacher_max_mean_entropy', 2.5)

        if teacher_mean_max_prob is not None and teacher_mean_max_prob < min_mean_max_prob:
            return True, (
                f'teacher mean max prob too low ({teacher_mean_max_prob:.3f} < {min_mean_max_prob:.3f})'
            )

        if teacher_mean_entropy is not None and teacher_mean_entropy > max_mean_entropy:
            return True, (
                f'teacher mean entropy too high ({teacher_mean_entropy:.3f} > {max_mean_entropy:.3f})'
            )

    if args.__dict__.get('teacher_filter_repeated_words', False):
        words = _word_sequence(teacher_pred_text)
        longest_repeat, longest_repeat_word = _longest_consecutive_repeat(words)
        max_word_repeat = args.__dict__.get('teacher_max_consecutive_word_repeat', 3)
        if longest_repeat > max_word_repeat:
            return True, (
                f'teacher word "{longest_repeat_word}" repeated {longest_repeat} times consecutively '
                f'(limit {max_word_repeat})'
            )

    if args.__dict__.get('teacher_filter_ctc_agreement', False) and ctc_text is not None:
        min_similarity = args.__dict__.get('teacher_ctc_agreement_min_similarity', 0.5)
        similarity = _sequence_similarity(_word_sequence(teacher_pred_text), _word_sequence(ctc_text))
        if similarity < min_similarity:
            return True, (
                f'encoder-decoder/ctc agreement too low ({similarity:.2f} < {min_similarity:.2f}); '
                f'ctc="{ctc_text}"'
            )

    return False, ''
