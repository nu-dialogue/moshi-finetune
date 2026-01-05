import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm


def merge_text_audio(
    text_ids: np.ndarray, audio_ids: np.ndarray, text_padding_id: int
) -> np.ndarray:
    """
    Merge the tokenized text and audio stream of a single speaker.
    Args:
        text_ids: Tokenized text stream. Shape: [T_text]
        audio_ids: Tokenized audio stream. Shape: [K=8, T_audio]
        text_padding_id: Padding id for text stream to fill the gap between audio and text streams.
    Returns:
        Merged tokenized text and audio stream. Shape: [K=8+1, T_audio]
    """
    assert text_ids.ndim == 1, f"Expected 1D tensor, got {text_ids.ndim}D tensor."
    assert audio_ids.ndim == 2, f"Expected 2D tensor, got {audio_ids.ndim}D tensor."
    # pad the text stream to match the audio stream
    audio_len = audio_ids.shape[1]

    out = np.empty((1 + audio_ids.shape[0], audio_len), dtype=np.int32)

    out[0].fill(np.int32(text_padding_id))
    n = min(text_ids.shape[0], audio_len)
    out[0, :n] = text_ids[:n].astype(np.int32, copy=False)

    if audio_ids.dtype != np.int32:
        out[1:] = audio_ids.astype(np.int32, copy=False)
    else:
        out[1:] = audio_ids

    return out.tolist()


def _list_dialogue_names(dir_path: str) -> list[str]:
    names = []
    with os.scandir(dir_path) as it:
        for e in it:
            if e.is_file() and e.name.endswith(".npz"):
                names.append(os.path.splitext(e.name)[0])
    return names


# --- multiprocessing worker ---
_WORK = {}


def _init_worker(
    tokenized_text_dir: str, tokenized_audio_dir: str, output_prefix: str, text_padding_id: int
):
    _WORK["text_dir"] = tokenized_text_dir
    _WORK["audio_dir"] = tokenized_audio_dir
    _WORK["out_prefix"] = output_prefix
    _WORK["pad"] = text_padding_id


def _process_one(dialogue_name: str):
    text_path = os.path.join(_WORK["text_dir"], f"{dialogue_name}.npz")
    audio_path = os.path.join(_WORK["audio_dir"], f"{dialogue_name}.npz")

    try:
        text_npz = np.load(text_path, allow_pickle=False)
        audio_npz = np.load(audio_path, allow_pickle=False)

        return {
            "dialogue_id": os.path.join(_WORK["out_prefix"], dialogue_name),
            "A": merge_text_audio(text_npz["A"], audio_npz["A"], _WORK["pad"]),
            "B": merge_text_audio(text_npz["B"], audio_npz["B"], _WORK["pad"]),
        }
    except Exception as e:
        print(f"Error processing {dialogue_name}: {e}")
        return None


def main(args):
    text_names = _list_dialogue_names(args.tokenized_text_dir)
    audio_names = _list_dialogue_names(args.tokenized_audio_dir)

    text_set = set(text_names)
    audio_set = set(audio_names)

    missing_text = audio_set - text_set
    missing_audio = text_set - audio_set

    if missing_text:
        print(f"Missing tokenized text for {len(missing_text)} dialogues.")
        with open("missing_text_dialogue_names.txt", "w") as f:
            f.write("\n".join(sorted(missing_text)))
    if missing_audio:
        print(f"Missing tokenized audio for {len(missing_audio)} dialogues.")
        with open("missing_audio_dialogue_names.txt", "w") as f:
            f.write("\n".join(sorted(missing_audio)))

    if not args.ignore_missing and (missing_text or missing_audio):
        print("Both text and audio tokenized dialogues should match.")
        return

    # only keep common dialogues (important if ignore_missing=True)
    dialogue_names = sorted(text_set & audio_set) if args.ignore_missing else sorted(text_names)

    out_dir = os.path.dirname(args.output_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    num_dialogues = len(dialogue_names)
    num_parquets = -(-num_dialogues // args.num_examples_per_parquet)

    max_workers = args.num_workers or os.cpu_count() or 1

    for i in range(num_parquets):
        chunk = dialogue_names[
            i * args.num_examples_per_parquet : (i + 1) * args.num_examples_per_parquet
        ]

        # load the tokenized text and audio data
        data = []
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_init_worker,
            initargs=(
                args.tokenized_text_dir,
                args.tokenized_audio_dir,
                args.output_prefix,
                args.text_padding_id,
            ),
        ) as ex:
            futures = [ex.submit(_process_one, name) for name in chunk]
            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Processing parquet {i + 1}/{num_parquets}",
            ):
                res = fut.result()
                if res is not None:
                    data.append(res)

        # save the merged data
        df = pd.DataFrame(data)
        output_path = f"{args.output_prefix}-{i + 1:03d}-of-{num_parquets:03d}.parquet"
        df.to_parquet(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge the tokenized text and audio data into a single dataset in parquet format."
    )
    parser.add_argument(
        "--ignore-missing",
        action="store_true",
        help="Ignore data which doesn't have both audio and text tokens. Print the number of ignored data.",
    )
    parser.add_argument(
        "--tokenized_text_dir",
        type=str,
        required=True,
        help="Path to the directory containing the tokenized text data.",
    )
    parser.add_argument(
        "--tokenized_audio_dir",
        type=str,
        required=True,
        help="Path to the directory containing the tokenized audio data.",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        required=True,
        help=(
            "Prefix for the output dataset. Output files will be named as "
            "`{{output_prefix}}-001-of-002.parquet` etc."
        ),
    )
    parser.add_argument(
        "--text_padding_id",
        type=int,
        default=3,
        help="Padding id for text stream to fill the gap between audio and text streams.",
    )
    parser.add_argument(
        "--num_examples_per_parquet",
        type=int,
        default=100_000,
        help="Number of samples per parquet file.",
    )
    parser.add_argument("--num_workers", type=int, default=0, help="0 = auto")
    args = parser.parse_args()

    main(args)
