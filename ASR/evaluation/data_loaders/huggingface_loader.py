"""
Data loader for HuggingFace datasets.
Downloads audio and ground truth locally for compatibility with TranscriptionPipeline.
"""

import os
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm
import pandas as pd

from .base_loader import BaseDataLoader


class HuggingFaceDataLoader(BaseDataLoader):
    """
    Data loader for HuggingFace datasets.

    Downloads audio files and saves ground truth locally for use with
    TranscriptionPipeline without modifications.

    Features:
    - Downloads audio from HuggingFace datasets
    - Saves as individual .wav files with original dataset IDs
    - Generates CSV with audio_id, audio_path, transcription, split
    - Compatible with datasets that have audio as path or waveform

    Example:
        >>> loader = HuggingFaceDataLoader(
        ...     dataset_name="jlvdoorn/atco2-asr",
        ...     audio_column="audio",
        ...     text_column="text"
        ... )
        >>> audio_dir, gt_csv = loader.download_and_save("./hf_cache")
        >>> gt = loader.load_ground_truth(gt_csv)
    """

    def __init__(
        self,
        dataset_name: str,
        audio_column: str = "audio",
        text_column: str = "text",
        split: Union[str, List[str], None] = "both",
        cache_dir: Optional[str] = None,
        id_column: Optional[str] = None,
    ):
        """
        Initialize HuggingFace data loader.

        Args:
            dataset_name: HuggingFace dataset name (e.g., "jlvdoorn/atco2-asr")
            audio_column: Column name containing audio data
            text_column: Column name containing transcription text
            split: Which split(s) to download - "train", "test", "both", or list ["train", "test"]
            cache_dir: Base directory for caching (default: ~/.cache/asr_hf_datasets)
            id_column: Optional column name for unique IDs (uses index if not provided)
        """
        self.dataset_name = dataset_name
        self.audio_column = audio_column
        self.text_column = text_column
        self.split = split
        self.id_column = id_column

        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.cache/asr_hf_datasets")
        self.cache_dir = Path(cache_dir)

        # Dataset will be loaded on first use
        self._dataset = None

    def id(self) -> str:
        """Unique identifier for this loader."""
        safe_name = self.dataset_name.replace("/", "_")
        return f"huggingface_{safe_name}"

    def _get_splits_to_load(self) -> List[str]:
        """Determine which splits to load based on split parameter."""
        if self.split == "both":
            return ["train", "test"]
        elif isinstance(self.split, str):
            return [self.split]
        elif isinstance(self.split, list):
            return self.split
        else:
            return ["train"]

    def _load_dataset(self):
        """Lazy load the dataset from HuggingFace."""
        if self._dataset is not None:
            return self._dataset

        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "The 'datasets' library is required for HuggingFaceDataLoader. "
                "Install it with: pip install datasets"
            )

        splits = self._get_splits_to_load()
        dataset = {}

        for split_name in splits:
            try:
                dataset[split_name] = load_dataset(
                    self.dataset_name,
                    split=split_name
                )
            except Exception as e:
                print(f"Warning: Could not load split '{split_name}': {e}")
                continue

        self._dataset = dataset
        return dataset

    def download_and_save(
        self,
        cache_dir: Optional[str] = None,
        overwrite: bool = False
    ) -> Tuple[str, str]:
        """
        Download dataset and save audio files and ground truth CSV locally.

        Args:
            cache_dir: Directory to save files (overrides constructor value)
            overwrite: If True, re-download even if files exist

        Returns:
            Tuple of (audio_directory_path, ground_truth_csv_path)

        Raises:
            ImportError: If 'datasets' library is not installed
            ValueError: If dataset cannot be loaded
        """
        if cache_dir:
            base_dir = Path(cache_dir)
        else:
            base_dir = self.cache_dir

        # Create directory structure: cache_dir/{dataset_name}/{split}/
        safe_name = self.dataset_name.replace("/", "_")
        dataset_dir = base_dir / safe_name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        gt_csv_path = dataset_dir / "ground_truth.csv"

        # Check if already downloaded
        if not overwrite and gt_csv_path.exists():
            # Verify CSV has content
            with open(gt_csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                headers = next(reader, None)
                if headers:
                    audio_dir = dataset_dir / "audio"
                    return str(audio_dir), str(gt_csv_path)

        # Load dataset
        dataset = self._load_dataset()
        if not dataset:
            raise ValueError(f"Could not load any splits from dataset '{self.dataset_name}'")

        # Prepare CSV data
        csv_rows = []

        for split_name, split_data in dataset.items():
            # Create audio directory for this split
            audio_dir = dataset_dir / "audio" / split_name
            audio_dir.mkdir(parents=True, exist_ok=True)

            print(f"Processing {split_name} split ({len(split_data)} examples)...")

            for idx, example in enumerate(tqdm(split_data, desc=f"Saving {split_name}")):
                # Get audio ID
                if self.id_column and self.id_column in example:
                    audio_id = str(example[self.id_column])
                else:
                    audio_id = f"{split_name}_{idx:06d}"

                # Get transcription
                if self.text_column not in example:
                    print(f"Warning: {self.text_column} not found in example {idx}, skipping")
                    continue
                transcription = str(example[self.text_column])

                # Process audio
                audio_path = None
                if self.audio_column in example:
                    audio_data = example[self.audio_column]
                    audio_path = self._save_audio(audio_data, audio_dir, audio_id)

                if audio_path:
                    csv_rows.append({
                        "audio_id": audio_id,
                        "audio_path": str(audio_path),
                        "transcription": transcription,
                        "split": split_name,
                    })

        # Save ground truth CSV using pandas
        if csv_rows:
            df = pd.DataFrame(csv_rows)
            df.to_csv(gt_csv_path, index=False, encoding='utf-8')
        else:
            # Create empty CSV with headers
            df = pd.DataFrame(columns=["audio_id", "audio_path", "transcription", "split"])
            df.to_csv(gt_csv_path, index=False, encoding='utf-8')

        audio_dir = dataset_dir / "audio"
        print(f"Saved {len(csv_rows)} examples to {dataset_dir}")
        return str(audio_dir), str(gt_csv_path)

    def _save_audio(self, audio_data, audio_dir: Path, audio_id: str) -> Optional[Path]:
        """
        Save audio data to .wav file.

        Handles audio as:
        - dict with 'path' and 'array' keys (HuggingFace datasets format)
        - dict with just 'path' key
        - string path
        - numpy array (waveform)

        Args:
            audio_data: Audio data in various formats
            audio_dir: Directory to save audio
            audio_id: Unique ID for this audio

        Returns:
            Path to saved .wav file, or None if failed
        """
        output_path = audio_dir / f"{audio_id}.wav"

        # Already exists
        if output_path.exists():
            return output_path

        # Handle None audio data
        if audio_data is None:
            print(f"Warning: Audio data is None for {audio_id}")
            return None

        try:
            # Handle torchcodec AudioDecoder (when torchcodec is installed)
            if hasattr(audio_data, 'get_all_samples'):
                # This is an AudioDecoder object from torchcodec
                samples = audio_data.get_all_samples()
                # Convert torch tensor to numpy array
                # samples.data has shape (num_channels, num_samples)
                audio_array = samples.data.numpy()
                # Handle both mono and stereo - take first channel if stereo
                if audio_array.ndim > 1:
                    audio_array = audio_array[0] if audio_array.shape[0] <= audio_array.shape[1] else audio_array[:, 0]
                # Get sample rate from the decoder
                sampling_rate = getattr(audio_data, 'sample_rate', 16000)
                sf.write(output_path, audio_array, sampling_rate)
                return output_path

            # HuggingFace datasets format: dict with path and array
            if isinstance(audio_data, dict):
                if 'array' in audio_data and 'sampling_rate' in audio_data:
                    # Save waveform
                    sf.write(output_path, audio_data['array'], audio_data['sampling_rate'])
                    return output_path
                elif 'path' in audio_data:
                    # Copy from existing path
                    import shutil
                    src_path = Path(audio_data['path'])
                    if src_path.exists():
                        shutil.copy2(src_path, output_path)
                        return output_path
                    else:
                        print(f"Warning: Source path does not exist: {src_path}")

            # String path
            elif isinstance(audio_data, str):
                import shutil
                src_path = Path(audio_data)
                if src_path.exists():
                    shutil.copy2(src_path, output_path)
                    return output_path
                else:
                    print(f"Warning: Source path does not exist: {src_path}")

            else:
                print(f"Warning: Unsupported audio data type: {type(audio_data)} for {audio_id}")

        except Exception as e:
            print(f"Warning: Could not save audio {audio_id}: {e}")
            import traceback
            traceback.print_exc()
            return None

        return None

    def load_ground_truth(self, data_path: str) -> Dict[str, str]:
        """
        Load ground truth from CSV file.

        Args:
            data_path: Path to ground truth CSV file

        Returns:
            Dictionary mapping {audio_id: transcription}
        """
        csv_path = Path(data_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Ground truth CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)
        ground_truth = {}

        for _, row in df.iterrows():
            audio_id = row.get("audio_id", "")
            transcription = row.get("transcription", "")
            if audio_id and transcription:
                ground_truth[audio_id] = transcription

        return ground_truth

    def load_ground_truth_by_split(self, data_path: str, split: str) -> Dict[str, str]:
        """
        Load ground truth for a specific split only.

        Args:
            data_path: Path to ground truth CSV file
            split: Split to filter by ("train" or "test")

        Returns:
            Dictionary mapping {audio_id: transcription} for specified split
        """
        csv_path = Path(data_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Ground truth CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)
        filtered_df = df[df["split"] == split]

        ground_truth = {}
        for _, row in filtered_df.iterrows():
            audio_id = row.get("audio_id", "")
            transcription = row.get("transcription", "")
            if audio_id and transcription:
                ground_truth[audio_id] = transcription

        return ground_truth

    def get_audio_path(self, ground_truth_id: str, audio_dir: str) -> Optional[str]:
        """
        Map ground truth ID to audio file path.

        Args:
            ground_truth_id: Audio ID from ground truth (e.g., "train_000001")
            audio_dir: Directory containing audio files (should contain train/ and test/ subdirs)

        Returns:
            Full path to audio file, or None if not found
        """
        audio_dir = Path(audio_dir)

        # Try to infer split from ID
        if ground_truth_id.startswith("train_"):
            wav_path = audio_dir / "train" / f"{ground_truth_id}.wav"
        elif ground_truth_id.startswith("test_"):
            wav_path = audio_dir / "test" / f"{ground_truth_id}.wav"
        else:
            # Try both splits
            wav_path = audio_dir / "train" / f"{ground_truth_id}.wav"
            if not wav_path.exists():
                wav_path = audio_dir / "test" / f"{ground_truth_id}.wav"

        if wav_path.exists():
            return str(wav_path)

        # Fallback: search recursively
        for wav_file in audio_dir.rglob(f"{ground_truth_id}.wav"):
            return str(wav_file)

        return None

    def get_dataset_info(self) -> Dict:
        """
        Get information about the loaded dataset.

        Returns:
            Dictionary with dataset statistics
        """
        dataset = self._load_dataset()

        info = {
            "dataset_name": self.dataset_name,
            "audio_column": self.audio_column,
            "text_column": self.text_column,
            "splits": {},
            "total_examples": 0,
        }

        for split_name, split_data in dataset.items():
            num_examples = len(split_data)
            info["splits"][split_name] = num_examples
            info["total_examples"] += num_examples

        return info
