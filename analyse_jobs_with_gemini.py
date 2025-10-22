#!/usr/bin/env python
import pandas as pd
from google import genai
import os
import time
import random
from typing import Dict, Any
import logging
import sys
import argparse
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class JobAnalyzer:
    def __init__(self, api_key: str = None, delay_between_requests: float = 5.0, thinking_budget: int = -1):
        if api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = genai.Client(api_key=os.environ.get('GEMINI_API_KEY'))
        
        self.model_name = 'gemini-2.5-flash'
        self.delay_between_requests = delay_between_requests
        self.thinking_budget = thinking_budget
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        
        # Move analysis instructions to system instruction for token efficiency
        self.system_instruction = """You are an expert job analyst.

Your task: Evaluate whether the job is matched to the typical skills and qualifications of a PhD graduate in computational physics, computational chemistry, or computational materials science:
- **Thoroughly review the job's required qualifications, duties, and field of work.**
- **Step-by-step evaluation criteria:**
    - Technical skills alignment (programming, simulation, modeling, calculations)
    - Scientific domain relevance (physics, chemistry, materials science)
    - Typical soft skills match from PhD experience
    - Any domain-specific requirements that are outside or adjacent to the above 3 fields
- **REASONING MUST APPEAR BEFORE THE FINAL CLASSIFICATION DECISION.**  
    - NEVER begin with or insert the classification before completing the full reasoning.

Classify as:
- "Well Fits": Direct, strong alignment with expected skills & expertise
- "Promising": Substantial overlap with likely knowledge/skills, not perfect match
- "Does Not Fit": Insufficient alignment with background/expected skills

Output format: Valid JSON only
{"reasoning": "[VERY concise reasoning]", 
"CLASSIFICATION": "[Well Fits/Promising/Does Not Fit]"}
"""
        
        # Simplified user prompt template (much shorter now)
        self.analysis_prompt = """Job Details:
Title: {title}
Company: {company}
Location: {location}
Description: {description}
First Qualification: {first_qualification}"""

    def extract_job_info(self, row: pd.Series) -> Dict[str, Any]:
        return {
            'title': str(row.get('title', 'N/A')),
            'company': str(row.get('company', 'N/A')),
            'location': str(row.get('location', 'N/A')),
            'description': str(row.get('description', 'N/A')),
            'first_qualification': str(row.get('first_qualification', 'N/A'))
        }


    def analyze_job_with_retry(self, job_info: Dict[str, Any], max_retries: int = 5) -> tuple[str, str]:
        """Analyze job with exponential backoff for rate limiting"""
        for attempt in range(max_retries):
            try:
                prompt = self.analysis_prompt.format(**job_info)
                
                # Enable thinking mode with AI Studio format
                from google.genai import types
                
                # Configure thinking mode using AI Studio format
                thinking_config = None
                if self.thinking_budget == -1:
                    thinking_config = types.ThinkingConfig(thinking_budget=-1)
                    logger.info("Using dynamic thinking mode")
                else:
                    thinking_config = types.ThinkingConfig(thinking_budget=self.thinking_budget)
                    logger.info(f"Using fixed thinking budget: {self.thinking_budget} tokens")
                
                # Simplified content structure per API docs
                generate_content_config = types.GenerateContentConfig(
                    temperature=0.2, 
                    thinking_config=thinking_config,
                    system_instruction=self.system_instruction
                )
                
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,  # Simplified: direct string
                    config=generate_content_config
                )
                analysis_text = response.text
                
                # Track token usage if available
                try:
                    if hasattr(response, 'usage_metadata'):
                        usage = response.usage_metadata
                        input_tokens = getattr(usage, 'prompt_token_count', 0)
                        output_tokens = getattr(usage, 'candidates_token_count', 0)
                        thinking_tokens = getattr(usage, 'thoughts_token_count', 0)
                        self.total_input_tokens += input_tokens
                        self.total_output_tokens += output_tokens
                        logger.info(f"Tokens used - Input: {input_tokens}, Output: {output_tokens}, Thinking: {thinking_tokens}")
                        logger.info(f"Total tokens - Input: {self.total_input_tokens}, Output: {self.total_output_tokens}")
                except Exception as token_error:
                    logger.debug(f"Could not track tokens: {token_error}")
                
                # Extract classification from the JSON response
                import re
                
                classification = "Unknown"
                try:
                    # Try to extract JSON from the response
                    json_match = re.search(r'\{[^}]*"CLASSIFICATION"[^}]*\}', analysis_text, re.DOTALL)
                    if json_match:
                        json_str = json_match.group()
                        parsed = json.loads(json_str)
                        classification = parsed.get("CLASSIFICATION", "Unknown")
                    else:
                        # Fallback: look for classification patterns
                        if '"CLASSIFICATION": "Well Fits"' in analysis_text:
                            classification = "Well Fits"
                        elif '"CLASSIFICATION": "Promising"' in analysis_text:
                            classification = "Promising"
                        elif '"CLASSIFICATION": "Does Not Fit"' in analysis_text:
                            classification = "Does Not Fit"
                        else:
                            logger.warning(f"Could not extract classification from response for job")
                            classification = "Unknown"
                except Exception as e:
                    logger.warning(f"Error parsing JSON response: {e}")
                    # Fallback parsing
                    if "Well Fits" in analysis_text:
                        classification = "Well Fits"
                    elif "Promising" in analysis_text:
                        classification = "Promising"
                    elif "Does Not Fit" in analysis_text:
                        classification = "Does Not Fit"
                    else:
                        classification = "Unknown"
                
                return analysis_text, classification
                
            except Exception as e:
                error_str = str(e)
                logger.warning(f"Attempt {attempt + 1} failed: {error_str}")
                
                # Check if it's a rate limit error
                if "429" in error_str or "quota" in error_str.lower() or "rate" in error_str.lower():
                    if attempt < max_retries - 1:
                        # Exponential backoff with jitter
                        wait_time = (2 ** attempt) * 60 + random.uniform(0, 30)  # Start with 60s + jitter
                        logger.info(f"Rate limit hit. Waiting {wait_time:.1f} seconds before retry...")
                        time.sleep(wait_time)
                        continue
                
                # Check if it's a service unavailable error
                elif "503" in error_str or "unavailable" in error_str.lower():
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) * 30 + random.uniform(0, 15)  # Start with 30s + jitter
                        logger.info(f"Service unavailable. Waiting {wait_time:.1f} seconds before retry...")
                        time.sleep(wait_time)
                        continue
                
                # For other errors, don't retry as much
                elif attempt < 2:
                    wait_time = 10 + random.uniform(0, 5)
                    logger.info(f"General error. Waiting {wait_time:.1f} seconds before retry...")
                    time.sleep(wait_time)
                    continue
                
                # If we've exhausted retries
                logger.error(f"Failed after {max_retries} attempts: {e}")
                return f"Error: {str(e)}", "Error"
        
        return "Error: Max retries exceeded", "Error"

    def process_jobs(self, input_file: str, output_file: str, start_index: int = 0):
        # Read the CSV
        logger.info(f"Reading CSV file: {input_file}")
        df = pd.read_csv(input_file)
        
        # Add new columns if they don't exist
        if 'gemini_analysis' not in df.columns:
            df['gemini_analysis'] = ''
        if 'gemini_classification' not in df.columns:
            df['gemini_classification'] = ''
        
        total_jobs = len(df)
        logger.info(f"Total jobs to process: {total_jobs}")
        logger.info(f"Starting from index: {start_index}")
        
        logger.info("Serial processing")
        self.process_jobs_serial(df, output_file, start_index)

    def process_jobs_serial(self, df: pd.DataFrame, output_file: str, start_index: int = 0):
        """Serial processing"""
        total_jobs = len(df)
        
        # Process each job starting from start_index
        for idx in range(start_index, total_jobs):
            if idx % 10 == 0:
                logger.info(f"Processing job {idx + 1}/{total_jobs}")
            
            # Skip if already processed
            if pd.notna(df.iloc[idx]['gemini_classification']) and df.iloc[idx]['gemini_classification'] != '':
                logger.info(f"Job {idx + 1} already processed, skipping")
                continue
            
            row = df.iloc[idx]
            job_info = self.extract_job_info(row)
            
            logger.info(f"Analyzing job {idx + 1}: {job_info['title']} at {job_info['company']}")
            
            analysis_text, classification = self.analyze_job_with_retry(job_info)
            
            # Update the dataframe
            df.iloc[idx, df.columns.get_loc('gemini_analysis')] = analysis_text
            df.iloc[idx, df.columns.get_loc('gemini_classification')] = classification
            
            logger.info(f"Job {idx + 1} classified as: {classification}")
            
            # Save progress every 10 jobs
            if (idx + 1) % 10 == 0:
                logger.info(f"Saving progress at job {idx + 1}")
                df.to_csv(output_file, index=False)
            
            # Rate limiting - configurable delay between API calls
            if self.delay_between_requests > 0:
                time.sleep(self.delay_between_requests)
        
        # Final save
        logger.info("Saving final results")
        df.to_csv(output_file, index=False)
        
        # Print summary
        classification_counts = df['gemini_classification'].value_counts()
        logger.info("Analysis complete!")
        logger.info("Classification summary:")
        for classification, count in classification_counts.items():
            logger.info(f"  {classification}: {count}")
        
        # Print token usage summary
        logger.info(f"\nToken Usage Summary:")
        logger.info(f"  Total Input Tokens: {self.total_input_tokens}")
        logger.info(f"  Total Output Tokens: {self.total_output_tokens}")
        logger.info(f"  Total Tokens: {self.total_input_tokens + self.total_output_tokens}")
        
        # Estimate cost (rough estimate for 2.5 Flash)
        estimated_cost = (self.total_input_tokens * 0.000075 / 1000) + (self.total_output_tokens * 0.0003 / 1000)
        logger.info(f"  Estimated Cost: ${estimated_cost:.4f}")  

def create_parser():
    """Create and configure the argument parser"""
    parser = argparse.ArgumentParser(
        description='Job analysis using Gemini 2.5 Flash with system instructions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - analyze jobs.csv, output to jobs_analyzed.csv
  python analyze_jobs_with_gemini.py jobs.csv
  
  # Fast processing (reduced delay for thousands of jobs)
  python analyze_jobs_with_gemini.py jobs.csv --delay 5
  
  # Disable thinking mode for speed
  python analyze_jobs_with_gemini.py jobs.csv --thinking-budget 0
  
  # Resume from specific index
  python analyze_jobs_with_gemini.py jobs.csv --start-index 100
  
  # Custom output file
  python analyze_jobs_with_gemini.py jobs.csv --output custom_results.csv
        """)
    
    parser.add_argument('input_file', 
                       help='Input CSV file containing job data')
    
    parser.add_argument('--output', '-o',
                       help='Output CSV file (default: [input]_analyzed.csv)')
    
    parser.add_argument('--start-index', '-s',
                       type=int, default=0,
                       help='Starting index for processing (default: 0)')
    
    parser.add_argument('--delay', '-d',
                       type=float, default=60.0,
                       help='Delay between requests in seconds (default: 60.0)')
    
    parser.add_argument('--thinking-budget', '-t',
                       type=int, default=-1,
                       help='Thinking budget in tokens (-1 for dynamic, 0 for disabled, positive for fixed, default: -1)')
    
    return parser

def main():
    parser = create_parser()
    args = parser.parse_args()
    
    # Check for API key
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        print("Error: Please set the GEMINI_API_KEY environment variable")
        print("You can get an API key from: https://aistudio.google.com/app/apikey")
        print("\nTo set the API key, run:")
        print("export GEMINI_API_KEY='your-api-key-here'")
        sys.exit(1)
    
    # Determine input and output files
    input_file = args.input_file
    if args.output:
        output_file = args.output
    else:
        # Generate output filename: input.csv -> input_analyzed.csv
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_analyzed.csv"
    
    # Validate input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' does not exist")
        sys.exit(1)
    
    print(f"Starting job analysis with Gemini 2.5 Flash...")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    
    print(f"\nConfiguration:")
    print(f"  Starting index: {args.start_index}")
    print(f"  Processing mode: Serial")
    print(f"  Delay between requests: {args.delay} seconds")
    thinking_mode = "Dynamic" if args.thinking_budget == -1 else "Disabled" if args.thinking_budget == 0 else f"Fixed ({args.thinking_budget} tokens)"
    print(f"  Thinking mode: {thinking_mode}")
    
    # Estimate time and cost based on input file
    try:
        df = pd.read_csv(input_file)
        total_jobs = len(df)
        remaining_jobs = max(0, total_jobs - args.start_index)
        
        print(f"  Total jobs in file: {total_jobs}")
        print(f"  Remaining jobs: {remaining_jobs}")
        
        estimated_time = remaining_jobs * args.delay / 60
        estimated_cost = remaining_jobs * 0.02  #  estimate
        print(f"  Estimated time: {estimated_time:.1f} minutes")
        print(f"  Estimated cost: ~${estimated_cost:.2f}")
        
    except Exception as e:
        print(f"  Could not estimate time/cost: {e}")
    
    # Initialize analyzer and run
    analyzer = JobAnalyzer(
        api_key, 
        delay_between_requests=args.delay,
        thinking_budget=args.thinking_budget
    )
    analyzer.process_jobs(input_file, output_file, args.start_index)

if __name__ == "__main__":
    main()
