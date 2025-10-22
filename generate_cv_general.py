#!/usr/bin/env python
"""
A simple script to tailor a LaTeX CV to a job description using an LLM.
"""
import argparse
import csv
import json
import os
import re
from typing import Dict, List, Optional, Tuple

# Assuming this script is in `gen/` and `utils` is a sibling directory.
from utils.llm_client import call_gemini

# --- Core Functions ---

def read_file(path: str) -> str:
    """Reads a file and returns its content."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def write_file(path: str, content: str):
    """Writes content to a file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def extract_latex_parts(latex_content: str) -> Tuple[str, str]:
    """
    Splits a LaTeX document into its preamble and body.

    Returns:
        A tuple containing (preamble, body_content).
        The `body_content` is the text between \begin{document} and \end{document}.
    """
    match = re.search(r"\\begin{document}(.*)\\end{document}", latex_content, re.DOTALL)
    if not match:
        raise ValueError("Could not find a \\begin{document}...\\end{document} block.")

    body_content = match.group(1).strip()
    preamble = latex_content[:match.start()]
    return preamble, body_content

def build_prompt(job_description: str, cv_body: str) -> List[Dict[str, str]]:
    """Builds the prompt for the LLM."""
    system_prompt = (
        "You are an expert career advisor editing a candidate's LaTeX CV. "
        "The user will provide the body of their CV and a job description. "
        "Rewrite the CV body to be perfectly tailored for the provided job description, "
        "focusing on highlighting the most relevant skills and experiences. "
        "Return a JSON object with the rewritten CV body under the key \"cv_body\". "
        "Ensure the output is valid LaTeX."
    )
    
    user_prompt = f"""
Here is the job description:
---
{job_description}
---

Here is the current LaTeX CV body:
---
{cv_body}
---

Please rewrite the CV body to align with the job description.
"""
    return [
        {"role": "developer", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

def main(argv: Optional[List[str]] = None) -> int:
    """Main function to run the tailoring process."""
    parser = argparse.ArgumentParser(description="Tailor a LaTeX CV to a job description.")
    parser.add_argument("--cv-in", default="source.tex", help="Path to the source LaTeX CV file.")
    parser.add_argument("--cv-out", default="tailored.tex", help="Path to write the tailored LaTeX CV file.")
    parser.add_argument("--csv-path", required=True, help="Path to the CSV file containing job descriptions.")
    parser.add_argument("--job-row", type=int, default=0, help="Row index of the job to use from the CSV (default: 0).")
    parser.add_argument("--desc-col", default="description", help="Name of the column with the job description (default: 'description').")
    parser.add_argument("--model", default="gemini-2.5-pro", help="The model to use for tailoring.")
    args = parser.parse_args(argv)

    print(f"Reading source CV from: {args.cv_in}")
    print(f"Reading job description from CSV: {args.csv_path} (row: {args.job_row})")

    # 1. Read files
    try:
        latex_content = read_file(args.cv_in)
        
        # Read job description from CSV
        with open(args.csv_path, mode='r', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            jobs = list(reader)
            if args.job_row >= len(jobs):
                print(f"Error: --job-row {args.job_row} is out of bounds. The CSV has {len(jobs)} jobs.")
                return 1
            
            job = jobs[args.job_row]
            if args.desc_col not in job:
                print(f"Error: Column '{args.desc_col}' not found in the CSV.")
                print(f"Available columns: {list(job.keys())}")
                return 1
            job_description = job[args.desc_col]

    except FileNotFoundError as e:
        print(f"Error: {e}. Please make sure the input files exist.")
        return 1
    except Exception as e:
        print(f"An error occurred while reading the CSV: {e}")
        return 1

    # 2. Extract LaTeX parts
    try:
        preamble, cv_body = extract_latex_parts(latex_content)
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    # 3. Build the prompt and call the LLM
    print("Asking the AI to tailor the CV... (this may take a moment)")
    messages = build_prompt(job_description, cv_body)
    
    # Ensure API key is set
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY environment variable is not set.")
        return 1

    try:
        response_text, usage = call_gemini(
            messages=messages,
            model=args.model,
            temperature=0.3,
            max_tokens=None,
        )
        
        # 4. Parse the response and write the output
        response_data = json.loads(response_text)
        tailored_body = response_data.get("cv_body")

        if not tailored_body:
            print("Error: The AI response did not contain the expected 'cv_body' key.")
            print("--- AI Response ---")
            print(response_text)
            print("---------------------")
            return 1

        final_latex = preamble + "\\begin{document}" + tailored_body + "\\end{document}\n"
        
        write_file(args.cv_out, final_latex)
        print(f"Successfully wrote tailored CV to: {args.cv_out}")

    except Exception as e:
        print(f"An error occurred during the AI call or processing: {e}")
        return 1

    return 0

if __name__ == "__main__":
    raise SystemExit(main())