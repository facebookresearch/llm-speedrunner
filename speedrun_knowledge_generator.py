import os
import json
import subprocess
from typing import Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path

from core.llm_client import LLMClient


@dataclass
class SpeedrunRecord:
    record_id: int
    code: str
    changelog: str
    next_code: Optional[str] = None


class SpeedrunAnalyzer:
    def __init__(
        self,
        records_dir: str,
        changelogs_dir: str,
        model_url: str,
        model_name: str = "deepseek_r1",
        api_key: str = "token-abc123"
    ):
        self.records_dir = Path(records_dir)
        self.changelogs_dir = Path(changelogs_dir)
        self.llm = LLMClient(
            model_url=model_url,
            model_name=model_name,
            log_metrics=True,
            api_key=api_key
        )
        
        # Load all records
        self.records: List[SpeedrunRecord] = []
        for i in range(1, 21):  # Records 1-21
            record_path = self.records_dir / f"record_{i}" / "train_gpt2.py"
            changelog_path = self.changelogs_dir / f"record_{i+1}.md" # +1 because the changelog is for the next record
            
            if not record_path.exists() or not changelog_path.exists():
                continue
                
            with open(record_path, "r") as f:
                code = f.read()
            with open(changelog_path, "r") as f:
                changelog = f.read()
                
            # Get next record's code if it exists
            next_record_path = self.records_dir / f"record_{i+1}" / "train_gpt2.py"
            next_code = None
            if next_record_path.exists():
                with open(next_record_path, "r") as f:
                    next_code = f.read()
                    
            self.records.append(SpeedrunRecord(
                record_id=i,
                code=code,
                changelog=changelog,
                next_code=next_code
            ))

    def generate_level_0(self, record: SpeedrunRecord) -> str:
        """Generate exact code diff to next record using git diff."""
        if not record.next_code:
            return "No next record available"
            
        # Create temporary files for git diff
        with open("temp_current.py", "w") as f:
            f.write(record.code)
        with open("temp_next.py", "w") as f:
            f.write(record.next_code)
            
        try:
            # Run git diff with no-index option to compare files
            # Don't use check=True since git diff returns non-zero when differences are found
            result = subprocess.run(
                ["git", "diff", "--no-index", "temp_current.py", "temp_next.py"],
                capture_output=True,
                text=True
            )
            return result.stdout
        except Exception as e:
            return f"Error generating diff: {str(e)}"
        finally:
            # Clean up temporary files
            if os.path.exists("temp_current.py"):
                os.remove("temp_current.py")
            if os.path.exists("temp_next.py"):
                os.remove("temp_next.py")

    def generate_level_1(self, record: SpeedrunRecord) -> str:
        """Generate pseudo code of the next record based on the git diff."""
        if not record.next_code:
            return "No next record available"
            
        # Get the git diff first
        diff = self.generate_level_0(record)
        if diff == "No next record available":
            return diff
            
        prompt = f"""Given the git diff between the current and next version and the changelog, generate a high-level pseudo code description of the changes made.
Focus on explaining the key algorithmic changes and improvements in a clear, concise way.

Git diff:
{diff}

Changelog:
{record.changelog}

Generate pseudo code that:
1. Describes the key algorithmic changes and improvements
2. Focuses on the high-level logic and avoids implementation details
3. Explains the purpose and impact of each major change
4. Uses clear, readable pseudo code syntax

Format the output as:
# Pseudo Code Changes
[Your pseudo code description here]"""

        return self.llm.generate(prompt)

    def generate_level_2(self, record: SpeedrunRecord) -> str:
        """Generate detailed natural language description of improvements."""
        prompt = f"""Given the current code, changelog, and next code, provide a detailed natural language description of the improvements made.
Current code:
{record.code}

Changelog:
{record.changelog}

Next code:
{record.next_code}

Provide a detailed explanation of:
1. What specific improvements were made
2. Why these changes were beneficial
3. How they contribute to the overall performance
4. Any technical challenges that were addressed"""

        return self.llm.generate(prompt)

    def generate_level_3(self, record: SpeedrunRecord) -> str:
        """Generate short-form hints in Twitter thread style."""
        prompt = f"""Given the current code, changelog, and next code, generate a concise Twitter thread-style summary of the key improvements.
Current code:
{record.code}

Changelog:
{record.changelog}

Next code:
{record.next_code}

Generate 3-5 short, engaging tweets that highlight the key improvements and their impact.
Use hashtags and emojis where appropriate.
Keep each tweet under 280 characters."""

        return self.llm.generate(prompt)

    def analyze_all_records(self, output_dir: str):
        """Generate all levels of analysis for all records and save to files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for record in self.records:
            record_dir = output_dir / f"record_{record.record_id}"
            record_dir.mkdir(exist_ok=True)
            
            # Generate and save all levels
            levels = {
                "level_0_diff.txt": self.generate_level_0(record),
                "level_1_pseudo.txt": self.generate_level_1(record),
                "level_2_description.txt": self.generate_level_2(record),
                "level_3_tweets.txt": self.generate_level_3(record)
            }
            
            for filename, content in levels.items():
                with open(record_dir / filename, "w") as f:
                    f.write(content)
                    
        # Flush LLM logs
        self.llm.flush_logs(output_dir / "llm_logs.jsonl")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze nanogpt speedrun records")
    parser.add_argument("--records_dir", required=True, help="Directory containing record_*.py files")
    parser.add_argument("--changelogs_dir", required=True, help="Directory containing changelog_*.txt files")
    parser.add_argument("--output_dir", required=True, help="Directory to save analysis results")
    parser.add_argument("--model_url", required=True, help="URL of the LLM server")
    parser.add_argument("--model_name", default="deepseek_r1", help="Name of the model to use")
    parser.add_argument("--api_key", default="token-abc123", help="API key for the LLM server")
    
    args = parser.parse_args()
    
    analyzer = SpeedrunAnalyzer(
        records_dir=args.records_dir,
        changelogs_dir=args.changelogs_dir,
        model_url=args.model_url,
        model_name=args.model_name,
        api_key=args.api_key
    )
    
    # import ipdb; ipdb.set_trace()
    analyzer.analyze_all_records(args.output_dir) 