import os
import sys
import random
import string
from datetime import datetime
from pathlib import Path
import pyperclip  # Add this import at the top with other imports

# Configuration
OUTPUT_DIRECTORY = Path("/Users/raghuvar/Code/vol_modeling_tmp/")  # <-- Change this to your desired output directory

def generate_random_string(length=8):
    """Generate a random alphanumeric string."""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def create_prompt_file(file_paths):
    """Create a prompt file from the given list of file paths."""
    # Ensure the output directory exists
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    # Generate filename
    random_str = generate_random_string()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_filename = f"{random_str}_{timestamp}.txt"
    output_path = OUTPUT_DIRECTORY / output_filename

    combined_content = ""  # Variable to store all content
    try:
        with output_path.open('w', encoding='utf-8') as outfile:
            for file_path in file_paths:
                path = Path(file_path)
                if path.is_file():
                    # Get immediate parent directory name
                    parent = path.parent.name
                    # If the file is in the root directory (no parent), handle accordingly
                    if parent == Path(path).anchor.strip(os.sep):
                        display_name = path.name
                    else:
                        display_name = f"{parent}/{path.name}"
                    
                    file_content = f"{display_name}:\n\n"
                    with path.open('r', encoding='utf-8') as infile:
                        contents = infile.read()
                        file_content += contents + "\n\n"
                        outfile.write(file_content)
                        combined_content += file_content  # Add to combined content
                else:
                    print(f"Warning: {file_path} is not a valid file and will be skipped.")
        
        # Copy combined content to clipboard
        pyperclip.copy(combined_content)
        print(f"Prompt file created at: {output_path}")
        print("Content has been copied to clipboard!")
    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python create_prompt.py <file1> <file2> ... <fileN>")
        sys.exit(1)
    
    file_paths = sys.argv[1:]
    create_prompt_file(file_paths)

if __name__ == "__main__":
        main()
