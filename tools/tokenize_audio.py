import argparse
import multiprocessing as mp
import os

import numpy as np
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from moshi.models import MimiModel, loaders
from tqdm import tqdm


def ceil(x, y):
    return int(-(-x // y))


def tokenize_audio(
    wav: torch.Tensor,
    mimi: MimiModel,
    audio_chunk_size: int,
) -> torch.LongTensor:
    """
    Tokenize the audio of a single channel.
    """
    assert wav.dim() == 1, f"Expected 1D tensor, got {wav.dim()}D tensor."

    wav_chunk_size = audio_chunk_size * mimi.sample_rate
    num_chunks = ceil(wav.shape[0], wav_chunk_size)
    device = next(mimi.parameters()).device

    list_of_audio_ids = []
    for i in range(num_chunks):
        wav_chunk = wav[i * wav_chunk_size : (i + 1) * wav_chunk_size]
        with torch.no_grad():
            list_of_audio_ids.append(
                mimi.encode(wav_chunk.reshape(1, 1, -1).to(device)).cpu()  # [B=1, K=8, T_chunk]
            )
    audio_ids = torch.cat(list_of_audio_ids, dim=-1)  # [B=1, K=8, T]
    audio_ids = audio_ids[0]  # [K=8, T]

    num_frames = ceil(wav.shape[-1], (mimi.sample_rate / mimi.frame_rate))
    assert audio_ids.shape == (
        mimi.num_codebooks,
        num_frames,
    ), f"{audio_ids.shape} != ({mimi.num_codebooks}, {num_frames})"
    return audio_ids


def worker(worker_id: int, gpu_id: int, dialogue_names: list[str], args: argparse.Namespace):
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    mimi = loaders.get_mimi(
        filename=hf_hub_download(args.audio_tokenizer_repo, args.audio_tokenizer_name),
        device=device,
    )
    pbar = tqdm(dialogue_names, desc=f"Worker {worker_id}", dynamic_ncols=True, position=worker_id)

    for dialogue_name in pbar:
        pbar.set_postfix_str(dialogue_name)

        # load audio
        wavs, sr = torchaudio.load(os.path.join(args.audio_dir, f"{dialogue_name}.wav"))
        assert wavs.shape[0] == 2, f"Expected stereo audio, got {wavs.shape[0]} channels."
        resampler = torchaudio.transforms.Resample(sr, mimi.sample_rate).to(device)
        wavs = resampler(wavs.to(device))

        # tokenize audio
        audio_ids_A = tokenize_audio(wavs[0], mimi, args.audio_chunk_size)
        audio_ids_B = tokenize_audio(wavs[1], mimi, args.audio_chunk_size)

        # save tokenized audio
        output_path = os.path.join(args.output_dir, f"{dialogue_name}.npz")
        np.savez_compressed(output_path, A=audio_ids_A.numpy(), B=audio_ids_B.numpy())


def main(args):
    mp.set_start_method("spawn", force=True)  # CUDA-safe

    dialogue_names = [
        os.path.splitext(d)[0] for d in os.listdir(args.audio_dir) if d.endswith(".wav")
    ]

    os.makedirs(args.output_dir, exist_ok=True)
    if args.resume:
        tokenized_dialogue_names = [
            os.path.splitext(d)[0] for d in os.listdir(args.output_dir) if d.endswith(".npz")
        ]
        print(f"Skipping {len(tokenized_dialogue_names)} already tokenized dialogues.")
        dialogue_names = list(set(dialogue_names) - set(tokenized_dialogue_names))

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No CUDA GPUs found.")

    dialogue_names_per_worker = np.array_split(dialogue_names, args.num_workers)

    processes: list[mp.Process] = []
    for worker_id, names in enumerate(dialogue_names_per_worker):
        gpu_id = worker_id % num_gpus
        p = mp.Process(target=worker, args=(worker_id, gpu_id, list(names), args))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tokenize audio files using a pretrained audio tokenizer."
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        required=True,
        help=(
            "Path to the directory containing the stereo wav files. "
            "Left and right channels should be the audio of speaker A and B respectively. "
            "and filenames should be the same as the dialogue names in the word transcript directory."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the directory to save the tokenized data.",
    )
    parser.add_argument(
        "--audio_tokenizer_repo",
        type=str,
        default="kyutai/moshiko-pytorch-bf16",
        help="Hugging Face Hub repository for the audio tokenizer.",
    )
    parser.add_argument(
        "--audio_tokenizer_name",
        type=str,
        default="tokenizer-e351c8d8-checkpoint125.safetensors",
        help="Model name for the audio tokenizer.",
    )

    parser.add_argument(
        "--audio_chunk_size",
        type=int,
        default=1200,
        help="Split audio into chunks of this size (seconds) to fit into cuda memory.",
    )
    parser.add_argument(
        "--num_workers", type=int, default=1, help="Number of workers for multiprocessing."
    )
    parser.add_argument("--resume", action="store_true", help="Resume tokenization.")
    args = parser.parse_args()

    main(args)
