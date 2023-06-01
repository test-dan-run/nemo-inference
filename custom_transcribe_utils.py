import json
from typing import List, Tuple, Union

from omegaconf import DictConfig
from nemo.collections.asr.parts.utils import rnnt_utils
from nemo.utils import logging

def normalize_timestamp_output(timestamps: dict):
    """
    Normalize the dictionary of timestamp values to JSON serializable values.
    Expects the following keys to exist -
        "start_offset": int-like object that represents the starting index of the token
            in the full audio after downsampling.
        "end_offset": int-like object that represents the ending index of the token
            in the full audio after downsampling.

    Args:
        timestamps: Nested dict.

    Returns:
        Normalized `timestamps` dictionary (in-place normalized)
    """
    for val_idx in range(len(timestamps)):
        timestamps[val_idx]['start_offset'] = int(timestamps[val_idx]['start_offset'])
        timestamps[val_idx]['end_offset'] = int(timestamps[val_idx]['end_offset'])
    return timestamps

def write_transcription(
    transcriptions: Union[List[rnnt_utils.Hypothesis], List[List[rnnt_utils.Hypothesis]]],
    cfg: DictConfig,
    model_name: str,
    filepaths: List[str] = None,
    compute_langs: bool = False,
    compute_timestamps: bool = False,
    compute_confidence: bool = False,
) -> Tuple[str, str]:
    """ Write generated transcription to output file. """
    if cfg.append_pred:
        logging.info(f'Transcripts will be written in "{cfg.output_filename}" file')
        if cfg.pred_name_postfix is not None:
            pred_by_model_name = cfg.pred_name_postfix
        else:
            pred_by_model_name = model_name
        pred_text_attr_name = 'pred_text_' + pred_by_model_name
    else:
        pred_text_attr_name = 'pred_text'

    if isinstance(transcriptions[0], rnnt_utils.Hypothesis):  # List[rnnt_utils.Hypothesis]
        best_hyps = transcriptions
        assert cfg.decoding.beam.return_best_hypothesis, "Works only with return_best_hypothesis=true"
    elif isinstance(transcriptions[0], list) and isinstance(
        transcriptions[0][0], rnnt_utils.Hypothesis
    ):  # List[List[rnnt_utils.Hypothesis]] NBestHypothesis
        best_hyps, beams = [], []
        for hyps in transcriptions:
            best_hyps.append(hyps[0])
            if not cfg.decoding.beam.return_best_hypothesis:
                beam = []
                for hyp in hyps:
                    beam.append((hyp.text, hyp.score))
                beams.append(beam)
    else:
        raise TypeError

    with open(cfg.output_filename, 'w', encoding='utf-8', newline='\n') as f:
        if cfg.audio_dir is not None:
            for idx, transcription in enumerate(best_hyps):  # type: rnnt_utils.Hypothesis
                item = {'audio_filepath': filepaths[idx], pred_text_attr_name: transcription.text}

                if compute_timestamps:
                    timestamps = transcription.timestep
                    if timestamps is not None and isinstance(timestamps, dict):
                        timestamps.pop('timestep', None)  # Pytorch tensor calculating index of each token, not needed.
                        for key in timestamps.keys():
                            values = normalize_timestamp_output(timestamps[key])
                            item[f'timestamps_{key}'] = values

                if compute_langs:
                    item['pred_lang'] = transcription.langs
                    item['pred_lang_chars'] = transcription.langs_chars
                if not cfg.decoding.beam.return_best_hypothesis:
                    item['beams'] = beams[idx]

                if compute_confidence:
                    item['token_confidence'] = transcription.token_confidence
                    item['word_confidence'] = transcription.word_confidence
                
                f.write(json.dumps(item) + "\n")
        else:
            with open(cfg.dataset_manifest, 'r', encoding='utf-8') as fr:
                for idx, line in enumerate(fr):
                    item = json.loads(line)
                    item[pred_text_attr_name] = best_hyps[idx].text

                    if compute_timestamps:
                        timestamps = best_hyps[idx].timestep
                        if timestamps is not None and isinstance(timestamps, dict):
                            timestamps.pop(
                                'timestep', None
                            )  # Pytorch tensor calculating index of each token, not needed.
                            for key in timestamps.keys():
                                values = normalize_timestamp_output(timestamps[key])
                                item[f'timestamps_{key}'] = values

                    if compute_langs:
                        item['pred_lang'] = best_hyps[idx].langs
                        item['pred_lang_chars'] = best_hyps[idx].langs_chars

                    if not cfg.decoding.beam.return_best_hypothesis:
                        item['beams'] = beams[idx]

                    if compute_confidence:
                        item['token_confidence'] = best_hyps[idx].token_confidence
                        item['word_confidence'] = best_hyps[idx].word_confidence

                    f.write(json.dumps(item) + "\n")

    return cfg.output_filename, pred_text_attr_name