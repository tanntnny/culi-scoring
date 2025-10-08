from __future__ import annotations

from pathlib import Path
from typing import Dict
import shutil

from huggingface_hub import snapshot_download

from ..interfaces.protocol import BaseDownloader
from ..core.registry import register

# Common processor/tokenizer files to extract
PROCESSOR_FILES = {
    "processor_config.json",
    "preprocessor_config.json", 
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.json",
    "vocab.txt",
    "config.json",
}

class HuggingFaceDownloader(BaseDownloader):
    """
    Simple Hugging Face downloader with easy configuration.
    
    Config format:
    items:
      - name: wav2vec2-base
        repo: facebook/wav2vec2-base
        type: model  # model | tokenizer | processor
        revision: main
        save: wav2vec2
    """
    
    def __init__(self, cfg):
        self.cfg = cfg
        
    def download(self):
        """Download all items specified in configuration"""
        print(f"Starting Hugging Face downloads...")
        
        if not hasattr(self.cfg.download, 'items') or not self.cfg.download.items:
            print("No items specified in configuration")
            return
        
        base_save_dir = Path(self.cfg.download.get('save_dir', 'models'))
        base_save_dir.mkdir(parents=True, exist_ok=True)
        
        for item in self.cfg.download.get("items", []):
            self._download_item(item, base_save_dir)
        
        print("All downloads completed!")
    
    def _download_item(self, item: Dict, base_save_dir: Path):
        """Download a single item"""
        name = item['name']
        repo = item['repo']
        item_type = item.get('type', 'model')
        revision = item.get('revision', 'main')
        save_path = item.get('save', name)
        
        print(f"Downloading {name} ({item_type}) from {repo}")
        
        # Create save directory
        save_dir = base_save_dir / save_path
        save_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            if item_type == 'model':
                self._download_model(repo, save_dir, revision)
            elif item_type == 'tokenizer':
                self._download_tokenizer(repo, save_dir, revision)
            elif item_type == 'processor':
                self._download_processor(repo, save_dir, revision)
            else:
                print(f"Unknown type '{item_type}', downloading as model")
                self._download_model(repo, save_dir, revision)
            
            print(f"Successfully downloaded {name}")
            
        except Exception as e:
            print(f"Failed to download {name}: {e}")
    
    def _download_model(self, repo: str, save_dir: Path, revision: str = 'main'):
        """Download complete model"""
        snapshot_download(
            repo_id=repo,
            local_dir=str(save_dir),
            revision=revision,
            ignore_patterns=["*.git*", "*.md", "*.txt", "README*"],
        )
    
    def _download_tokenizer(self, repo: str, save_dir: Path, revision: str = 'main'):
        """Download only tokenizer files"""
        # First try to download the whole model then extract tokenizer files
        temp_dir = save_dir / "temp"
        temp_dir.mkdir(exist_ok=True)
        
        try:
            # Download full model to temp
            snapshot_download(
                repo_id=repo,
                local_dir=str(temp_dir),
                revision=revision,
            )
            
            # Copy only tokenizer files
            copied_files = []
            for filename in PROCESSOR_FILES:
                src = temp_dir / filename
                if src.exists():
                    shutil.copy2(src, save_dir / filename)
                    copied_files.append(filename)
            
            print(f"Extracted tokenizer files: {copied_files}")
            
        finally:
            # Clean up temp directory
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
    
    def _download_processor(self, repo: str, save_dir: Path, revision: str = 'main'):
        """Download processor files (same as tokenizer for most cases)"""
        self._download_tokenizer(repo, save_dir, revision)


@register("downloader", "hf")
def build_hf_downloader(cfg):
    """Build Hugging Face downloader"""
    return HuggingFaceDownloader(cfg)