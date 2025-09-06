"""
Ghost Logger - Autonomous Research Documentation System (CORRECTED VERSION)
Auto-captures everything when you hit 'play' in your IDE with zero input needed.

PHILOSOPHY:
- Same test script = Same directory (logical grouping)
- Multiple runs = Versioned content within that directory
- Output files named with run numbers for distinction
- Complete evolution tracking in one place
"""

import os
import sys
import datetime
import subprocess
import json
import ast
import platform
import psutil
import functools
import contextlib
import io
import traceback
import hashlib
import shutil
import glob
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import inspect

class GhostLogger:
    def __init__(self):
        self.base_dir = Path("research_logs")
        self.base_dir.mkdir(exist_ok=True)
        self.timeline_file = self.base_dir / "timeline.md"
        self.current_session = {}
        self.start_time = None
        self.terminal_buffer = io.StringIO()
        self.pre_execution_files = set()
        self.execution_dir = None
        
    def blockchain_timestamp(self, content: str = "") -> str:
        """Generate tamper-proof blockchain-style hash timestamp for IP protection."""
        utc_time = datetime.datetime.utcnow()
        local_time = datetime.datetime.now()
        
        timezone_name = time.tzname[time.daylight] if time.daylight else time.tzname[0]
        readable_timestamp = f"{local_time.strftime('%Y-%m-%d %H:%M:%S')} {timezone_name}"
        
        utc_iso = utc_time.isoformat() + "Z"
        system_info = f"{platform.node()}{platform.processor()}{psutil.virtual_memory().total}"
        
        hash_input = f"{utc_iso}{content}{system_info}{os.getpid()}"
        tamper_proof_hash = hashlib.sha256(hash_input.encode()).hexdigest()
        
        return f"{readable_timestamp}:{tamper_proof_hash[:16]}"
    
    def create_ip_proof_hash(self, script_content: str, execution_data: str) -> str:
        """Create a comprehensive IP protection hash."""
        utc_time = datetime.datetime.utcnow()
        local_time = datetime.datetime.now()
        timezone_name = time.tzname[time.daylight] if time.daylight else time.tzname[0]
        
        readable_timestamp = f"{local_time.strftime('%Y-%m-%d %H:%M:%S')} {timezone_name}"
        utc_iso = utc_time.isoformat() + "Z"
        system_fingerprint = f"{platform.node()}{platform.processor()}"
        
        full_content = f"{utc_iso}{script_content}{execution_data}{system_fingerprint}"
        ip_hash = hashlib.sha256(full_content.encode()).hexdigest()
        
        return f"IP-PROOF:{readable_timestamp}:SHA256:{ip_hash}"
    
    def extract_comment_tags(self, script_path: str) -> Dict[str, str]:
        """Extract # Summary:, # Comments:, # Expect:, # Notes: from script."""
        tags = {
            'summary': '',
            'comments': '', 
            'expect': '',
            'notes': ''
        }
        
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            for line in content.split('\n'):
                line = line.strip()
                if line.startswith('# Summary:'):
                    tags['summary'] = line[10:].strip()
                elif line.startswith('# Comments:'):
                    tags['comments'] = line[11:].strip()
                elif line.startswith('# Expect:'):
                    tags['expect'] = line[9:].strip()
                elif line.startswith('# Notes:'):
                    tags['notes'] = line[8:].strip()
                    
        except Exception as e:
            self.log_error(f"Error extracting tags: {e}")
            
        return tags
    
    def analyze_code(self, script_path: str) -> Dict[str, Any]:
        """Analyze Python script for functions, classes, imports."""
        analysis = {
            'functions': [],
            'classes': [],
            'imports': [],
            'variables': [],
            'errors': []
        }
        
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    analysis['functions'].append(node.name)
                elif isinstance(node, ast.ClassDef):
                    analysis['classes'].append(node.name)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        analysis['imports'].append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    for alias in node.names:
                        analysis['imports'].append(f"{module}.{alias.name}")
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            analysis['variables'].append(target.id)
                            
        except Exception as e:
            analysis['errors'].append(str(e))
            
        return analysis
    
    def detect_daily_subjects(self, date_str: str) -> str:
        """Auto-detect research subjects from script names run today."""
        subjects = set()
        for folder in self.base_dir.glob(f"{date_str}_*"):
            if folder.is_dir():
                parts = folder.name.split('_')[1:]
                subjects.update(parts)
        
        script_name = Path(sys.argv[0]).stem if sys.argv else "unknown"
        script_subjects = self.extract_subjects_from_name(script_name)
        subjects.update(script_subjects)
        
        if subjects:
            return f"{date_str}_" + "-".join(sorted(subjects))
        else:
            return f"{date_str}_exploration"
    
    def extract_subjects_from_name(self, script_name: str) -> List[str]:
        """Extract research subjects from script filename by parsing meaningful terms."""
        ignore_words = {
            'test', 'tests', 'testing', 'script', 'run', 'main', 'demo', 'example',
            'temp', 'tmp', 'old', 'new', 'backup', 'copy', 'final', 'draft',
            'debug', 'check', 'validate', 'analyze', 'analysis', 'exploration',
            'experiment', 'study', 'research', 'project', 'work', 'code'
        }
        
        import re
        name_parts = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)|[0-9]+', script_name)
        name_parts.extend(script_name.lower().replace('_', ' ').replace('-', ' ').split())
        
        subjects = []
        for part in name_parts:
            part = part.lower().strip()
            if len(part) >= 3 and part not in ignore_words and part.isalpha():
                subjects.append(part)
        
        subjects = list(set(subjects))[:4]
        return subjects if subjects else ['exploration']
    
    def get_system_info(self) -> Dict[str, str]:
        """Gather system information for documentation."""
        info = {
            'python_version': sys.version.split()[0],
            'platform': platform.platform(),
            'processor': platform.processor(),
            'architecture': platform.architecture()[0],
            'hostname': platform.node(),
        }
        
        try:
            info['memory_gb'] = f"{psutil.virtual_memory().total / (1024**3):.1f} GB"
            info['cpu_count'] = str(psutil.cpu_count())
        except:
            info['memory_gb'] = "Unknown"
            info['cpu_count'] = "Unknown"
            
        return info
    
    def create_test_directory(self, script_path: str) -> Path:
        """
        CORRECTED: Create/reuse directory for same test script.
        Same script = same directory, multiple runs tracked within.
        """
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        daily_folder = self.detect_daily_subjects(date_str)
        daily_path = self.base_dir / daily_folder
        daily_path.mkdir(exist_ok=True)
        
        # Create test directory based on FIRST run timestamp if new,
        # or reuse existing directory for same script
        script_name = Path(script_path).stem
        
        # Look for existing directory for this script
        existing_dirs = list(daily_path.glob(f"*_{script_name}"))
        
        if existing_dirs:
            # Reuse existing directory (same logical test)
            test_dir = existing_dirs[0]
        else:
            # Create new directory with current timestamp
            timestamp = datetime.datetime.now().strftime("%H-%M-%S")
            test_dir = daily_path / f"{timestamp}_{script_name}"
            test_dir.mkdir(exist_ok=True)
        
        # Ensure subdirectories exist
        (test_dir / "outputs").mkdir(exist_ok=True)
        (test_dir / "versions").mkdir(exist_ok=True)
        
        return test_dir
    
    def capture_pre_execution_state(self, script_path: str) -> None:
        """Capture the state of the working directory before execution."""
        self.execution_dir = Path(script_path).parent.absolute()
        
        self.pre_execution_files = set()
        for pattern in ['*', '*/*', '*/*/*']:
            self.pre_execution_files.update(
                str(p.absolute()) for p in self.execution_dir.glob(pattern) 
                if p.is_file()
            )
    
    def capture_and_move_new_files(self, test_dir: Path, run_number: int) -> List[str]:
        """
        CORRECTED: Find and move files with run number in filename.
        This way you can distinguish which run generated which files.
        """
        if not self.execution_dir:
            return []
        
        current_files = set()
        for pattern in ['*', '*/*', '*/*/*']:
            current_files.update(
                str(p.absolute()) for p in self.execution_dir.glob(pattern) 
                if p.is_file()
            )
        
        new_files = current_files - self.pre_execution_files
        moved_files = []
        
        outputs_dir = test_dir / "outputs"
        outputs_dir.mkdir(exist_ok=True)
        
        for file_path in new_files:
            try:
                file_path_obj = Path(file_path)
                if 'research_logs' in str(file_path_obj):
                    continue
                
                # Add run number to filename for distinction
                stem = file_path_obj.stem
                suffix = file_path_obj.suffix
                run_filename = f"{stem}_run{run_number}{suffix}"
                
                dest_path = outputs_dir / run_filename
                counter = 1
                while dest_path.exists():
                    run_filename = f"{stem}_run{run_number}_{counter}{suffix}"
                    dest_path = outputs_dir / run_filename
                    counter += 1
                
                shutil.move(str(file_path_obj), str(dest_path))
                moved_files.append(run_filename)
                
            except Exception as e:
                self.log_error(f"Error moving file {file_path}: {e}")
        
        return moved_files
    
    def create_version_backup(self, test_dir: Path, script_path: str, run_number: int) -> None:
        """Create versioned backup of script for this run."""
        versions_dir = test_dir / "versions"
        versions_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        script_name = Path(script_path).stem
        backup_name = f"{script_name}_v{run_number}_{timestamp}.py"
        backup_path = versions_dir / backup_name
        
        try:
            with open(script_path, 'r', encoding='utf-8') as src:
                with open(backup_path, 'w', encoding='utf-8') as backup:
                    backup.write(f"# VERSION BACKUP - Run #{run_number}\n")
                    backup.write(f"# Timestamp: {self.blockchain_timestamp()}\n") 
                    backup.write(f"# Original: {script_path}\n")
                    backup.write(f"# Hash: {self.get_file_hash(script_path)}\n")
                    backup.write(f"# {'='*50}\n\n")
                    backup.write(src.read())
                    
        except Exception as e:
            self.log_error(f"Error creating version backup: {e}")
    
    def detect_code_changes(self, script_path: str, test_dir: Path) -> Dict[str, str]:
        """Detect if code has changed since last run."""
        versions_dir = test_dir / "versions"
        if not versions_dir.exists():
            return {"status": "first_run", "details": "No previous versions to compare"}
            
        version_files = sorted(versions_dir.glob("*.py"), key=lambda x: x.stat().st_mtime, reverse=True)
        if not version_files:
            return {"status": "first_run", "details": "No previous versions found"}
            
        latest_backup = version_files[0]
        current_hash = self.get_file_hash(script_path)
        
        try:
            with open(latest_backup, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith("# Hash: "):
                        backup_hash = line.split("# Hash: ")[1].strip()
                        if current_hash != backup_hash:
                            return {
                                "status": "changed",
                                "details": f"Code modified since {latest_backup.name}",
                                "previous_hash": backup_hash,
                                "current_hash": current_hash
                            }
                        else:
                            return {
                                "status": "unchanged", 
                                "details": f"No changes since {latest_backup.name}"
                            }
        except Exception as e:
            self.log_error(f"Error checking code changes: {e}")
            
        return {"status": "unknown", "details": "Could not determine changes"}
    
    def get_run_number(self, test_dir: Path) -> int:
        """
        CORRECTED: Get next run number by counting existing runs.
        Look at versions folder and terminal output to determine current run.
        """
        # Check versions folder for existing runs
        versions_dir = test_dir / "versions"
        if versions_dir.exists():
            version_files = list(versions_dir.glob("*.py"))
            if version_files:
                return len(version_files) + 1
        
        # Check if research_log.md exists and count runs
        md_file = test_dir / "research_log.md"
        if md_file.exists():
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    import re
                    run_matches = re.findall(r'## Run #(\d+)', content)
                    if run_matches:
                        return max(int(m) for m in run_matches) + 1
            except Exception:
                pass
        
        return 1
    
    def create_research_manifest(self, test_dir: Path) -> None:
        """Create a manifest file listing all versions and their purposes."""
        versions_dir = test_dir / "versions" 
        manifest_path = test_dir / "research_manifest.txt"
        
        if not versions_dir.exists():
            return
            
        try:
            version_files = sorted(versions_dir.glob("*.py"), key=lambda x: x.stat().st_mtime)
            
            with open(manifest_path, 'w', encoding='utf-8') as manifest:
                manifest.write(f"# Research Evolution Manifest\n")
                manifest.write(f"# Test Directory: {test_dir.name}\n") 
                manifest.write(f"# Generated: {self.blockchain_timestamp()}\n")
                manifest.write(f"# {'='*50}\n\n")
                
                for i, version_file in enumerate(version_files, 1):
                    with open(version_file, 'r', encoding='utf-8') as vf:
                        lines = vf.readlines()[:10]
                        
                    timestamp = ""
                    hash_val = ""
                    run_num = ""
                    
                    for line in lines:
                        if "# Timestamp: " in line:
                            timestamp = line.split("# Timestamp: ")[1].strip()
                        elif "# Hash: " in line:
                            hash_val = line.split("# Hash: ")[1].strip()
                        elif "# VERSION BACKUP - Run #" in line:
                            run_num = line.split("Run #")[1].strip()
                            
                    manifest.write(f"Version {i}: {version_file.name}\n")
                    manifest.write(f"  - Run Number: #{run_num}\n")
                    manifest.write(f"  - Timestamp: {timestamp}\n")
                    manifest.write(f"  - Hash: {hash_val}\n")
                    manifest.write(f"  - Size: {version_file.stat().st_size} bytes\n")
                    manifest.write(f"\n")
                    
        except Exception as e:
            self.log_error(f"Error creating research manifest: {e}")
    
    def generate_research_md(self, test_dir: Path, script_path: str, 
                           analysis: Dict, tags: Dict, system_info: Dict,
                           run_number: int, execution_result: Dict, moved_files: List[str],
                           code_changes: Dict) -> None:
        """
        CORRECTED: Update existing research_log.md with new run information.
        Same document tracks all runs for this test script.
        """
        
        script_name = Path(script_path).stem.replace('_', ' ').title()
        
        script_content = self.read_file_safe(script_path)
        execution_data = str(execution_result)
        ip_proof = self.create_ip_proof_hash(script_content, execution_data)
        blockchain_ts = self.blockchain_timestamp(script_content + execution_data)
        
        md_path = test_dir / "research_log.md"
        is_update = md_path.exists()
        
        # Code change summary
        change_summary = ""
        if code_changes["status"] == "changed":
            change_summary = f"**Code Modified**: {code_changes['details']}"
        elif code_changes["status"] == "unchanged":
            change_summary = f"**Code Status**: No changes since last run"
        else:
            change_summary = f"**Code Status**: {code_changes['details']}"
        
        if is_update:
            # APPEND new run information to existing document
            with open(md_path, 'a', encoding='utf-8') as f:
                f.write(f"""

---

## Run #{run_number}

**Timestamp**: {blockchain_ts}  
**Status**: {execution_result.get('status', 'Unknown')}  
**Duration**: {execution_result.get('duration', 'Unknown')}  
**IP Protection Hash**: `{ip_proof}`

### Code Changes
{change_summary}

### Execution Summary
- **Result**: {'SUCCESS' if execution_result.get('success', False) else 'ERROR/EXCEPTION'}
- **Key Metrics**: {execution_result.get('summary', 'Execution completed')}

### Generated Files (Run #{run_number})
{self.format_file_list(moved_files) if moved_files else 'No files generated'}

### Notes
Run #{run_number} details available in `terminal_output.txt` and `versions/` folder.

""")
        else:
            # CREATE new research document
            template = f"""# {script_name}

**Research Subject**: {tags.get('summary', 'Automated Research Test')}

---

## Executive Summary
{tags.get('summary', 'Exploratory automated test')}

## Research Context

### Background
{tags.get('comments', 'Automated capture - no background provided')}

### Hypotheses & Expectations  
{tags.get('expect', 'No specific expectations provided')}

### Research Notes
{tags.get('notes', 'No additional notes provided')}

---

## Technical Environment

| Component | Details |
|-----------|---------|
| **Python Version** | {system_info['python_version']} |
| **Platform** | {system_info['platform']} |
| **Architecture** | {system_info['architecture']} |
| **Memory** | {system_info['memory_gb']} |
| **CPU Count** | {system_info['cpu_count']} |
| **Hardware** | {system_info['processor']} |

### Libraries Used
{', '.join(analysis.get('imports', ['None detected']))}

---

## Code Analysis

### Functions Detected
{', '.join(analysis.get('functions', ['None'])) if analysis.get('functions') else 'None detected'}

### Classes Detected  
{', '.join(analysis.get('classes', ['None'])) if analysis.get('classes') else 'None detected'}

---

## Execution History

### Run #{run_number}

**Timestamp**: {blockchain_ts}  
**Status**: {execution_result.get('status', 'Unknown')}  
**Duration**: {execution_result.get('duration', 'Unknown')}  
**Script Hash**: `{self.get_file_hash(script_path)}`  
**IP Protection Hash**: `{ip_proof}`

#### Code Changes
{change_summary}

#### Execution Results
- **Result**: {'SUCCESS' if execution_result.get('success', False) else 'ERROR/EXCEPTION'}
- **Key Metrics**: {execution_result.get('summary', 'Execution completed')}

#### Generated Files (Run #{run_number})
{self.format_file_list(moved_files) if moved_files else 'No files generated'}

---

## File Structure

| File | Purpose | Status |
|------|---------|--------|
| `research_log.md` | This documentation | ✓ All runs tracked |
| `code_snapshot.py` | Latest source code | ✓ Current |
| `terminal_output.txt` | All execution logs | ✓ All runs |
| `outputs/` | Generated files (with run numbers) | ✓ Organized by run |
| `versions/` | All code iterations | ✓ Complete history |
| `research_manifest.txt` | Version timeline | ✓ All versions |

---

## IP Protection Notice

All runs cryptographically timestamped for IP protection.
**Local Storage**: Complete privacy, no cloud backup.

**Generated**: {blockchain_ts}  
**Ghost Logger Version**: 2.1 (Corrected Run Handling)
"""

            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(template)
    
    def format_file_list(self, files: List[str]) -> str:
        """Format list of files for markdown display."""
        if not files:
            return "None generated"
        return '\n'.join([f"- `{file}`" for file in files])
    
    def read_file_safe(self, file_path: str) -> str:
        """Safely read file content."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception:
            return f"# Could not read file: {file_path}"
    
    def get_file_hash(self, file_path: str) -> str:
        """Generate hash of file for verification."""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()[:16]
        except Exception:
            return "unknown"
    
    def capture_terminal_output(self, test_dir: Path, output: str, run_number: int) -> None:
        """
        CORRECTED: Append terminal output with clear run separation.
        All runs in same file, clearly distinguished.
        """
        terminal_file = test_dir / "terminal_output.txt"
        blockchain_ts = self.blockchain_timestamp(output)
        
        mode = 'a' if terminal_file.exists() else 'w'
        with open(terminal_file, mode, encoding='utf-8') as f:
            if mode == 'a':
                f.write(f"\n\n{'='*60}\n")
                f.write(f"RUN #{run_number} - {blockchain_ts}\n")
                f.write(f"{'='*60}\n\n")
            else:
                f.write(f"Ghost Logger Terminal Capture\n")
                f.write(f"Test: {test_dir.name}\n")
                f.write(f"Started: {blockchain_ts}\n")
                f.write(f"{'='*60}\n")
                f.write(f"RUN #{run_number}\n")
                f.write(f"{'='*60}\n\n")
            
            for line in output.split('\n'):
                f.write(f"[{self.blockchain_timestamp(line)}] {line}\n")
    
    def save_code_snapshot(self, test_dir: Path, script_path: str) -> None:
        """Save snapshot of the current code (latest version)."""
        snapshot_file = test_dir / "code_snapshot.py"
        try:
            with open(script_path, 'r', encoding='utf-8') as src:
                with open(snapshot_file, 'w', encoding='utf-8') as dst:
                    dst.write(f"# Code Snapshot - {self.blockchain_timestamp()}\n")
                    dst.write(f"# Latest version of: {script_path}\n")
                    dst.write(f"# See versions/ folder for complete history\n\n")
                    dst.write(src.read())
        except Exception as e:
            self.log_error(f"Error saving code snapshot: {e}")
    
    def update_timeline(self, test_dir: Path, script_name: str, tags: Dict, run_number: int) -> None:
        """Update the global timeline."""
        blockchain_ts = self.blockchain_timestamp(f"{script_name}{run_number}")
        entry = f"- **{blockchain_ts}** - {script_name} (Run #{run_number}) - {tags.get('summary', 'Auto test')}\n"
        
        if not self.timeline_file.exists():
            with open(self.timeline_file, 'w', encoding='utf-8') as f:
                f.write("# Research Timeline\n\nChronological log of all tests and explorations.\n\n")
        
        with open(self.timeline_file, 'a', encoding='utf-8') as f:
            f.write(entry)
    
    def log_error(self, message: str) -> None:
        """Log error messages."""
        error_file = self.base_dir / "ghost_logger_errors.log"
        timestamp = self.blockchain_timestamp()
        with open(error_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] ERROR: {message}\n")

# Global instance
_ghost_logger = GhostLogger()

def capture_everything(func):
    """Decorator to auto-capture everything about a test function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        frame = inspect.currentframe()
        script_path = frame.f_back.f_globals.get('__file__', 'unknown_script.py')
        
        ghost = _ghost_logger
        ghost.start_time = datetime.datetime.now()
        
        # Same test script = same directory (logical grouping)
        test_dir = ghost.create_test_directory(script_path)
        run_number = ghost.get_run_number(test_dir)
        
        # Check for code changes and create version backup
        code_changes = ghost.detect_code_changes(script_path, test_dir)
        ghost.create_version_backup(test_dir, script_path, run_number)
        
        # Capture pre-execution state for output files
        ghost.capture_pre_execution_state(script_path)
        
        # Extract information
        tags = ghost.extract_comment_tags(script_path)
        analysis = ghost.analyze_code(script_path)
        system_info = ghost.get_system_info()
        
        # Save/update code snapshot
        ghost.save_code_snapshot(test_dir, script_path)
        
        # Capture execution
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        captured_output = io.StringIO()
        
        execution_result = {
            'success': False,
            'status': 'Started',
            'duration': '0s',
            'summary': 'Execution initiated'
        }
        
        try:
            sys.stdout = captured_output
            sys.stderr = captured_output
            
            result = func(*args, **kwargs)
            
            execution_result.update({
                'success': True,
                'status': 'Completed Successfully',
                'summary': 'Function executed without errors'
            })
            
        except Exception as e:
            execution_result.update({
                'success': False,
                'status': f'Error: {type(e).__name__}',
                'summary': f'Exception occurred: {str(e)}'
            })
            captured_output.write(f"\nEXCEPTION: {str(e)}\n")
            captured_output.write(traceback.format_exc())
            
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            
            duration = datetime.datetime.now() - ghost.start_time
            execution_result['duration'] = f"{duration.total_seconds():.2f}s"
            
            # Capture and move files with run number in filename
            moved_files = ghost.capture_and_move_new_files(test_dir, run_number)
            
            output = captured_output.getvalue()
            ghost.capture_terminal_output(test_dir, output, run_number)
            ghost.generate_research_md(test_dir, script_path, analysis, tags, 
                                     system_info, run_number, execution_result, moved_files, code_changes)
            ghost.update_timeline(test_dir, Path(script_path).stem, tags, run_number)
            ghost.create_research_manifest(test_dir)
            
            # Summary output
            print(f"\n{'='*60}")
            print(f"Ghost Logger: Run #{run_number} captured!")
            print(f"Directory: {test_dir}")
            print(f"Status: {execution_result['status']}")
            print(f"Duration: {execution_result['duration']}")
            if moved_files:
                print(f"Files: {', '.join(moved_files)}")
            print(f"Code: {code_changes['status']}")
            print(f"LOCAL STORAGE - All runs tracked in same logical folder")
            print(f"{'='*60}")
        
        return result if 'result' in locals() else None
    
    return wrapper

if __name__ == "__main__":
    print("Ghost Logger 2.1 - Same test = same directory, runs tracked within")