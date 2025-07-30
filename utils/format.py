import argparse
import concurrent.futures
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from tqdm import tqdm


# =========== CONFIGURATION FUNCTIONS ===========
def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Check or fix formatting of files.')
    parser.add_argument('--fix',
                        action='store_true',
                        help='Fix formatting issues instead of just checking')
    parser.add_argument('--changed-only',
                        action='store_true',
                        help='Only check files changed since last commit')
    parser.add_argument('--jobs',
                        '-j',
                        type=int,
                        default=os.cpu_count(),
                        help='Number of parallel jobs')
    parser.add_argument(
        '--show-errors',
        action='store_true',
        help='Show detailed formatting errors for files that fail checks')
    return parser.parse_args()


def setup_formatters(proj_dir):
    """Initialize formatter configurations and check availability."""
    # File type configurations
    formatters = {
        ".py": {
            "cmd": "yapf",
            "check_args": ["--diff", f"{proj_dir}/.style.yapf"],
            "fix_args": ["--in-place", f"{proj_dir}/.style.yapf"],
            "isort_check": ["isort", "--check-only", "--diff"],
            "isort_fix": ["isort"],
        },
        ".sh": {
            "cmd": "beautysh",
            "check_args": ["--check", "--indent-size", "2"],
            "fix_args": ["--indent-size", "2"],
        },
        ".md": {
            "cmd": "mdformat",
            "check_args": ["--check"],
            "fix_args": [],
        },
        ".yaml": {
            "cmd": "yamllint",
            "check_args": ["-f", "parsable", "-c", f"{proj_dir}/.yamllint"],
            "fix_args": ["-f", "parsable", "-c", f"{proj_dir}/.yamllint"],
        },
        ".yml": {
            "cmd": "yamllint",
            "check_args": ["-f", "parsable", "-c", f"{proj_dir}/.yamllint"],
            "fix_args": ["-f", "parsable", "-c", f"{proj_dir}/.yamllint"],
        },
    }

    # Check which formatters are available
    available_formatters = {}
    for ext, config in formatters.items():
        if shutil.which(config["cmd"]):
            available_formatters[ext] = config
        else:
            print(
                f"Warning: Formatter '{config['cmd']}' for {ext} files not found - skipping"
            )

    return formatters, available_formatters


# =========== FILE DISCOVERY FUNCTIONS ===========
def get_gitignore_patterns(directory):
    """Read and parse gitignore patterns from .gitignore file."""
    patterns = []
    gitignore_path = os.path.join(directory, '.gitignore')

    if os.path.exists(gitignore_path):
        with open(gitignore_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    patterns.append(line)

    return patterns


def build_exclusion_patterns(exclude_patterns):
    """
    Convert gitignore-style patterns into regex patterns.
    Returns a list of compiled regex patterns.
    """
    regex_patterns = []

    for pattern in exclude_patterns:
        pattern = pattern.strip()
        if not pattern or pattern.startswith('#'):
            continue

        # Handle directory-specific patterns (ending with /)
        is_dir_pattern = pattern.endswith('/')
        if is_dir_pattern:
            pattern = pattern[:-1]

        # Convert pattern to regex
        regex = re.escape(pattern).replace('\\*', '.*')

        # Apply pattern as appropriate based on whether it's directory or file pattern
        if is_dir_pattern:
            regex_patterns.append(re.compile(f'(^|/){regex}(/|$)'))
        else:
            regex_patterns.append(re.compile(f'(^|/){regex}$'))

    return regex_patterns


def is_path_excluded(path, base_dir, exclusion_patterns):
    """Check if a file or directory path matches any exclusion pattern."""
    if not exclusion_patterns:
        return False

    # Get relative path for matching
    rel_path = os.path.relpath(path, base_dir)

    # Check against all patterns
    for pattern in exclusion_patterns:
        if pattern.search(rel_path):
            return True

    return False


def discover_files_in_dir(directory, exclusion_patterns, formatters, base_dir):
    """Discover files in a single directory."""
    files_to_format = []
    files_without_formatter = []

    start_time = time.time()
    print(f"Discovering files in {directory}...")

    for root, dirs, filenames in os.walk(directory):
        # Skip excluded directories - modify dirs in-place to prevent traversal
        dirs[:] = [
            d for d in dirs if not is_path_excluded(os.path.join(
                root, d), base_dir, exclusion_patterns)
        ]

        for filename in filenames:
            filepath = os.path.join(root, filename)

            # Skip excluded files
            if is_path_excluded(filepath, base_dir, exclusion_patterns):
                continue

            # Track files by extension
            ext = os.path.splitext(filename)[1]
            path_obj = Path(filepath)

            if ext in formatters:
                files_to_format.append(path_obj)
            elif ext:  # Only report files with non-empty extensions
                files_without_formatter.append((ext, path_obj))

    elapsed = time.time() - start_time
    print(
        f"Found {len(files_to_format)} files to format in {directory} in {elapsed:.4f} seconds"
    )

    return files_to_format, files_without_formatter


def discover_files_parallel(directories, formatters, exclusion_patterns,
                            base_dir, max_workers):
    """Discover files in multiple directories in parallel."""
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(len(directories), max_workers)) as executor:
        futures = {
            executor.submit(discover_files_in_dir, d, exclusion_patterns, formatters, base_dir):
            d
            for d in directories
        }
        all_discovered = []
        all_without_formatter = []

        for future in concurrent.futures.as_completed(futures):
            try:
                files_to_format, files_without_formatter = future.result()
                all_discovered.extend(files_to_format)
                all_without_formatter.extend(files_without_formatter)
            except Exception as exc:
                print(f"Directory discovery generated an exception: {exc}")

    # Group files without formatters by extension for better reporting
    extensions_without_formatter = {}
    for ext, filepath in all_without_formatter:
        if ext not in extensions_without_formatter:
            extensions_without_formatter[ext] = []
        extensions_without_formatter[ext].append(filepath)

    # Report extensions without formatters
    if extensions_without_formatter:
        print("\nFiles with extensions that have no configured formatters:")
        for ext, files in extensions_without_formatter.items():
            file_count = len(files)
            print(f"  {ext}: {file_count} file(s)")
            # Print a few examples
            for i, f in enumerate(files[:3]):
                print(f"    - {f}")
            if file_count > 3:
                print(f"    - ... ({file_count-3} more)")

    return all_discovered


def get_subdirectories(src_dirs, exclusion_patterns, base_dir):
    """Get subdirectories for parallel file discovery."""
    subdirs = []
    for d in src_dirs:
        base_dir_path = Path(d)
        # Always include the base directory itself to ensure root files are processed
        subdirs.append(base_dir_path)

        # Get immediate subdirectories, excluding ones to skip
        try:
            immediate_subdirs = [
                base_dir_path / subdir.name
                for subdir in base_dir_path.iterdir() if subdir.is_dir()
                and not is_path_excluded(subdir, base_dir, exclusion_patterns)
            ]
            if immediate_subdirs:
                subdirs.extend(immediate_subdirs)
        except (PermissionError, FileNotFoundError) as e:
            print(f"Error accessing directory {base_dir_path}: {e}")

    return subdirs


# =========== GIT INTEGRATION FUNCTIONS ===========
def get_changed_files(proj_dir):
    """Get files that have changed according to git."""
    try:
        # Get files that have uncommitted changes
        result = subprocess.run(
            ["git", "-C", proj_dir, "diff", "--name-only", "HEAD"],
            capture_output=True,
            text=True,
            check=True)
        changed = [
            proj_dir / f for f in result.stdout.splitlines()
            if Path(proj_dir / f).exists()
        ]

        # Get staged files
        result = subprocess.run(
            ["git", "-C", proj_dir, "diff", "--name-only", "--staged"],
            capture_output=True,
            text=True,
            check=True)
        staged = [
            proj_dir / f for f in result.stdout.splitlines()
            if Path(proj_dir / f).exists()
        ]

        # Get untracked files
        result = subprocess.run([
            "git", "-C", proj_dir, "ls-files", "--others", "--exclude-standard"
        ],
                                capture_output=True,
                                text=True,
                                check=True)
        untracked = [
            proj_dir / f for f in result.stdout.splitlines()
            if Path(proj_dir / f).exists()
        ]

        all_changed = changed + staged + untracked
        return all_changed
    except (subprocess.SubprocessError, FileNotFoundError):
        print(
            "Warning: Could not get changed files from git, checking all files"
        )
        return None


# =========== FORMATTER EXECUTION FUNCTIONS ===========
def process_file(file_info):
    """Process a single file with the appropriate formatter."""
    f, extension, config, args = file_info

    # Special handling for YAML files since yamllint only validates
    if extension in [".yaml", ".yml"]:
        cmd = [config["cmd"]] + config["check_args"] + [f"{f}"]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                # yamllint found issues
                error_msg = f"YAML lint issues found in {extension} file"
                if args.show_errors:
                    error_msg += f"\n{r.stdout.strip()}"
                return (f, error_msg)
            return None
        except Exception as e:
            return (f, f"Error processing: {str(e)}")

    # Special handling for Python files: run isort before yapf
    if extension == ".py":
        # Check mode: run isort --check-only --diff
        if not args.fix:
            isort_cmd = config.get("isort_check") + [str(f)]
            try:
                isort_result = subprocess.run(isort_cmd,
                                              capture_output=True,
                                              text=True)
                if isort_result.returncode != 0:
                    error_msg = f"Import organization issues found in Python file"
                    if args.show_errors:
                        error_msg += f"\n{isort_result.stdout.strip()}"
                    return (f, error_msg)
            except Exception as e:
                return (f, f"Error running isort: {str(e)}")
        # Fix mode: run isort
        else:
            isort_cmd = config.get("isort_fix") + [str(f)]
            try:
                isort_result = subprocess.run(isort_cmd,
                                              capture_output=True,
                                              text=True)
                if isort_result.returncode != 0:
                    error_msg = f"Error fixing import organization in Python file"
                    if args.show_errors:
                        error_msg += f"\n{isort_result.stdout.strip()}"
                    return (f, error_msg)
            except Exception as e:
                return (f, f"Error running isort: {str(e)}")

    # Use fix_args if --fix is specified, otherwise use check_args
    cmd_args = config["fix_args"] if args.fix else config["check_args"]
    cmd = [config["cmd"]] + cmd_args + [f"{f}"]

    try:
        r = subprocess.run(cmd, capture_output=True, text=True)

        if not args.fix and (r.stdout or
                             (r.returncode != 0 and extension != ".md")):
            # In check mode, report unformatted files
            error_msg = f"Unformatted {extension} file"
            if args.show_errors:
                # Show stdout/stderr for more details
                details = r.stdout.strip() or r.stderr.strip()
                if details:
                    error_msg += f"\n{details}"
            return (f, error_msg)
        elif args.fix and r.returncode != 0:
            # In fix mode, report files with errors during formatting
            error_msg = f"Error formatting: {r.stderr.strip()}"
            if args.show_errors:
                details = r.stdout.strip()
                if details:
                    error_msg += f"\n{details}"
            return (f, error_msg)
        return None
    except Exception as e:
        return (f, f"Error processing: {str(e)}")


def process_files_parallel(files, available_formatters, formatters, args):
    """Process multiple files in parallel using available formatters."""
    unformatted_files = []

    print(f"Checking {len(files)} files...")

    # Prepare arguments for parallel processing
    file_tasks = []
    for f in files:
        extension = f.suffix
        if extension in available_formatters:
            file_tasks.append(
                (f, extension, available_formatters[extension], args))
        elif extension in formatters:
            unformatted_files.append(f)
            print(
                f"Skipped due to missing formatter: {formatters[extension]['cmd']}: {f}"
            )
        else:
            unformatted_files.append(f)
            print(f"No formatter configured for {extension} file: {f}")

    # Process files in parallel
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.jobs) as executor:
        futures = {
            executor.submit(process_file, task): task
            for task in file_tasks
        }

        for future in tqdm(concurrent.futures.as_completed(futures),
                           total=len(file_tasks),
                           desc="Formatting progress"):
            result = future.result()
            if result:
                f, message = result
                print(f"{message}: {f}")
                unformatted_files.append(f)

    return unformatted_files


# =========== MAIN FUNCTION ===========
def main():
    # Parse command line arguments
    args = parse_arguments()

    # Setup project directory and configurations
    proj_dir = Path(__file__).resolve().parents[1]
    src_dirs = [f"{proj_dir}"]

    # Define exclusion patterns (gitignore format)
    exclude_patterns = [
        "*.npy", "*.txt", ".git/", "__pycache__/", "node_modules/", "build/",
        "dist/"
    ]

    # Add patterns from gitignore
    gitignore_patterns = get_gitignore_patterns(proj_dir)
    exclude_patterns.extend(gitignore_patterns)

    # Build consolidated exclusion patterns
    exclusion_patterns = build_exclusion_patterns(exclude_patterns)

    # Setup formatters
    formatters, available_formatters = setup_formatters(proj_dir)

    # Get files to check
    all_files = []

    # Check changed files or discover all files
    if args.changed_only:
        changed_files = get_changed_files(proj_dir)
        if changed_files:
            print(f"Only checking {len(changed_files)} changed files")
            all_files = [f for f in changed_files if f.suffix in formatters]
        else:
            args.changed_only = False

    if not args.changed_only:
        # Get subdirectories for parallel processing
        subdirs = get_subdirectories(src_dirs, exclusion_patterns, proj_dir)
        all_files = discover_files_parallel(subdirs, formatters,
                                            exclusion_patterns, proj_dir,
                                            args.jobs)

    # Process files
    unformatted_files = process_files_parallel(all_files, available_formatters,
                                               formatters, args)

    # Report results
    if unformatted_files:
        if args.fix:
            print(f"Failed to fix {len(unformatted_files)} file(s)")
        else:
            print(
                f"Found {len(unformatted_files)} file(s) with formatting issues"
            )
            print("Run with --fix to automatically format the files")
        if args.show_errors:
            print("\nDetailed formatting errors:")
            for f in unformatted_files:
                # Each entry is either a Path or a tuple (Path, message)
                if isinstance(f, tuple):
                    path, msg = f
                    print(f"\n{path}:")
                    print(msg)
        sys.exit(1)
    else:
        if args.fix:
            print("All files formatted successfully")
        else:
            print("All files formatted correctly")
        sys.exit(0)


if __name__ == "__main__":
    main()
