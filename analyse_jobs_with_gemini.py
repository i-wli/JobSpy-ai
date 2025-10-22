#!/usr/bin/env python
import pandas as pd
from google import genai
import os
import time
import random
from typing import Dict, Any
import logging
import sys

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
        
        self.analysis_prompt = """Analyze and determine whether this job suitably matches the typical skills and qualifications of a new PhD graduate in computational physics, computational chemistry, or computational materials science.

First, carefully consider each job's required qualifications, duties, and field of work to see if it aligns with the background and expertise of such a candidate. Show your thinking process for every job, explaining how and why it does or does not fit before stating a clear classification ("Fits" or "Does Not Fit"). Do not make the classification before outlining your reasoning.

Job Details:
Title: {title}
Company: {company}
Location: {location}
Description: {description}
First Qualification: {first_qualification}

Please provide your analysis and end with exactly one of: "CLASSIFICATION: Fits" or "CLASSIFICATION: Does Not Fit"
"""

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
                
                # Enable thinking mode with dynamic budget and use 2.5 Flash
                from google.genai import types
                
                # Configure thinking mode
                thinking_config = None
                if self.thinking_budget == -1:
                    # Dynamic thinking - let the model decide how much thinking it needs
                    thinking_config = types.ThinkingConfig()
                    logger.info("Using dynamic thinking mode")
                else:
                    # Fixed thinking budget
                    thinking_config = types.ThinkingConfig(
                        max_thinking_tokens=self.thinking_budget
                    )
                    logger.info(f"Using fixed thinking budget: {self.thinking_budget} tokens")
                
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        thinking_config=thinking_config
                    )
                )
                analysis_text = response.text
                
                # Track token usage if available
                try:
                    if hasattr(response, 'usage_metadata'):
                        usage = response.usage_metadata
                        input_tokens = getattr(usage, 'prompt_token_count', 0)
                        output_tokens = getattr(usage, 'candidates_token_count', 0)
                        thinking_tokens = getattr(usage, 'thinking_token_count', 0)
                        self.total_input_tokens += input_tokens
                        self.total_output_tokens += output_tokens
                        logger.info(f"Tokens used - Input: {input_tokens}, Output: {output_tokens}, Thinking: {thinking_tokens}")
                        logger.info(f"Total tokens - Input: {self.total_input_tokens}, Output: {self.total_output_tokens}")
                except Exception as token_error:
                    logger.debug(f"Could not track tokens: {token_error}")
                
                # Extract classification from the response
                if "CLASSIFICATION: Fits" in analysis_text:
                    classification = "Fits"
                elif "CLASSIFICATION: Does Not Fit" in analysis_text:
                    classification = "Does Not Fit"
                else:
                    logger.warning("Could not extract classification from response")
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
            logger.info(f"Waiting {self.delay_between_requests} seconds before next request...")
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

def main():
    # Check for API key
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        print("Error: Please set the GEMINI_API_KEY environment variable")
        print("You can get an API key from: https://aistudio.google.com/app/apikey")
        print("\nTo set the API key, run:")
        print("export GEMINI_API_KEY='your-api-key-here'")
        sys.exit(1)
    
    print("Starting job analysis with Gemini 2.5 Flash (with thinking mode)...")
    print("This will analyze each job in filtered_jobs_tier5.csv")
    print("Results will be saved to filtered_jobs_tier5_analyzed.csv")
    print(f"\nNote: This will take a while as there are many jobs to process")
    print("The script saves progress every 10 jobs, so you can stop and restart if needed")
    
    # Get parameters from command line or user input
    start_index = 0
    delay = 5.0  # Default 5 seconds based on generous rate limits for 2.5 Flash
    thinking_budget = -1  # Default dynamic thinking (let model decide)
    
    if len(sys.argv) > 1:
        start_index = int(sys.argv[1])
    if len(sys.argv) > 2:
        delay = float(sys.argv[2])
    if len(sys.argv) > 3:
        thinking_budget = int(sys.argv[3])
    
    if len(sys.argv) <= 3:
        try:
            start_input = input(f"\nEnter starting index (0 to start from beginning): ").strip()
            if start_input:
                start_index = int(start_input)
            
            delay_input = input(f"Enter delay between requests in seconds (default 5 for generous rate limits): ").strip()
            if delay_input:
                delay = float(delay_input)
                
            thinking_input = input(f"Enter thinking budget in tokens (-1 for dynamic, or fixed number like 32768): ").strip()
            if thinking_input:
                thinking_budget = int(thinking_input)
        except (ValueError, KeyboardInterrupt):
            print(f"\nUsing defaults - start index: {start_index}, delay: {delay} seconds, thinking budget: {thinking_budget}")
    
    print(f"Starting analysis from index: {start_index}")
    print(f"Delay between requests: {delay} seconds")
    thinking_mode = "Dynamic (let model decide)" if thinking_budget == -1 else f"{thinking_budget} tokens"
    print(f"Thinking mode: {thinking_mode}")
    print(f"Estimated time for remaining jobs: {delay * 72 / 60:.1f} minutes")
    
    # Initialize analyzer and run
    input_file = 'filtered_jobs_tier5.csv'
    output_file = 'filtered_jobs_tier5_analyzed.csv'
    
    analyzer = JobAnalyzer(api_key, delay_between_requests=delay, thinking_budget=thinking_budget)
    analyzer.process_jobs(input_file, output_file, start_index)

if __name__ == "__main__":
    main()
