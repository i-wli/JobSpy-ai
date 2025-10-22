#!/usr/bin/env python
"""
Split job scraper - separates scraping from filtering/scoring.
Allows independent adjustment of filtering without re-scraping.

Enhanced scoring includes:
- 3x multiplier for degree_role/core_domain in first qualifications point
- 5x multiplier for negative_keywords in first qualifications point
"""

import csv
import json
import random
import re
import time
import os
from datetime import datetime
from math import ceil
from typing import Dict, Iterable, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import pandas as pd
from jobspy import scrape_jobs


# ------------------------------
# Configuration
# ------------------------------

# Job boards (LinkedIn included with conservative rate limiting)
SITES = ["indeed", "glassdoor", "google", "linkedin"]

# Per-combination result cap (per site, per search term, per country)
RESULTS_PER_COMBO = 50

# Recency window in hours (e.g., 336 = 14 days, 168 = 7 days)
HOURS_OLD = 168

# Desired description format from scrapers
DESCRIPTION_FORMAT = "markdown"

# Minimum score to keep a job
RELEVANCE_THRESHOLD = 4

# Site-specific rate limiting configuration (LinkedIn is most restrictive)
SITE_DELAYS = {
    "indeed": (3, 8),     # min, max delay in seconds
    "glassdoor": (5, 12), # Glassdoor is stricter 
    "google": (2, 6),     # Google tends to be more permissive
    "linkedin": (20, 35), # LinkedIn is very restrictive - even longer delays to avoid rate limits
}

# Checkpoint and recovery configuration
CHECKPOINT_DIR = "scraper_checkpoints"
CHECKPOINT_INTERVAL = 5  # Save progress every N combinations
MAX_PARALLEL_SITES = 4  # Number of sites to scrape simultaneously
RETRY_FAILED_AFTER = 300  # Retry failed combinations after N seconds

# Rate limit detection patterns
RATE_LIMIT_INDICATORS = [
    "rate limit", "too many requests", "429", "403", 
    "blocked", "captcha", "temporary", "try again later"
]

# Strategically ordered search terms - broad coverage first, then niche specialization
SEARCH_TERMS = [
    # Tier 1: Broad, high-coverage terms (capture ~60-70% of relevant jobs)
    "Research Scientist",           # Catches most research roles across domains
    "Data scientist",               # High-volume term, many relevant hits
    "Computational Scientist",      # Core domain, broad appeal
    "Applied Scientist",            # Industry research roles
    
    # Tier 2: Domain-specific terms (fill core domain gaps)
    "scientific computing",         # Broad computational science
    "computational chemistry",      # Core domain
    "computational physics",        # Core domain  
    "AI for science",              # Growing interdisciplinary field
    
    # Tier 3: Specialized application areas (targeted gaps)
    "machine learning materials science",  # Specific AI+materials
    "physics-informed machine learning",   # Specific AI+physics
    "quantum chemistry",                   # Specialized chemistry
    "molecular modeling",                  # Specialized chemistry
    "AI for materials",                    # Materials-focused AI
    
    # Tier 4: Highly specialized/niche terms (rare but valuable roles)
    "HPC Scientist",                      # High-performance computing
    "Scientific Software Engineer",        # Software development for science
    "materials informatics",             # Data science for materials
    "cheminformatics",                    # Data science for chemistry
    "molecular dynamics",                 # Specialized simulation
    "Research Software Engineer",         # Software engineering for research
    "Scientific Programmer",             # Programming for science
    "Modelling and Simulation Scientist", # Simulation specialist
    "Algorithm modeling Engineer",        # Algorithm development
    "AI drug discovery",                  # Pharma-specific AI
    "ML materials discovery",             # Materials discovery AI
    "AI in physics",                     # Physics-focused AI
    "DFT",                               # Very specialized quantum
    "ab initio",                         # Very specialized quantum
]

# Countries to search (expand as needed)
COUNTRIES = [
    "UK",
    "Germany",
    "Netherlands",
    "Switzerland",
    "Ireland",
    "Sweden",
    "Denmark",
    "Finland",
    "Belgium",
    "Austria",
]

# Broad search terms that need domain filtering to avoid irrelevant results
BROAD_TERMS_NEEDING_FILTER = {
    "Research Scientist",
    "Data scientist", 
    "Computational Scientist",
    "Applied Scientist"
}


def get_country_indeed(country: str) -> str:
    """Map display country names to JobSpy's Country enum strings for Indeed/Glassdoor.
    Country.from_string is case-insensitive and supports e.g. "uk,united kingdom".
    We pass canonical human-readable names and let JobSpy normalize them.
    """
    mapping = {
        "UK": "UK",
        "United Kingdom": "UK",
        "Germany": "Germany",
        "Netherlands": "Netherlands",
        "Switzerland": "Switzerland",
        "France": "France",
        "Ireland": "Ireland",
        "Sweden": "Sweden",
        "Denmark": "Denmark",
        "Finland": "Finland",
        "Spain": "Spain",
        "Italy": "Italy",
        "Belgium": "Belgium",
        "Austria": "Austria",
        "Norway": "Norway",
        "Portugal": "Portugal",
        "Czech Republic": "Czech Republic",
        "Poland": "Poland",
    }
    return mapping.get(country, country)

# ------------------------------
# Relevance Scoring System with Enhanced Qualifications Parsing
# ------------------------------

POSITIVE_KEYWORDS: Dict[str, Tuple[Iterable[str], int]] = {
    # group: (terms, weight)
    "degree_role": (
        [
            "phd",
            "doctorate",
            "doctoral",
            "research scientist",
            "postdoctoral",
            "postdoc",
        ],
        3,
    ),
    "core_domain": (
        [
            "theoretical chemistry",
            "computational chemistry",
            "quantum chemistry",
            "molecular modeling",
            "cheminformatics",
            "quantitative",
            "materials informatics",
            # "computational physics",
            "computational materials",
            "scientific computing",
            "molecular dynamics",
            "dft",
            "ab initio",
            "simulation",
            # "docking",
            # "free energy",
            # "enhanced sampling",
            "nature science",
            "physics",
            "quantitative",
        ],
        3,
    ),
    "methods": (
        [
            "machine learning",
            "deep learning",
            "foundation model",
            "physics-informed",
            "optimization",
            "active learning",
            "statistical modeling",
            "predictive modeling",
        ],
        2,
    ),
    "core_applications": (
        [
            "drug discovery",
            "materials discovery",
            "catalyst design",
            "materials science",
            "protein design",
            "structure prediction",
            "molecular design",
        ],
        2,
    ),
    "applications": (
        [
            "battery",
            "solar cells",
            "energy storage",
            "protein folding",
            "protein design",
            "polymer",
            "semiconductor",
            "nanomaterial",
        ],
        1,
    ),
    "tools_infra": (
        [
            "pytorch",
            "tensorflow",
            "jax",
            "scikit-learn",
            "rdkit",
            "openmm",
            "gromacs",
            "lammps",
            "vasp",
            "gaussian",
            "orca",
            "cuda",
            "mpi",
            "hpc",
        ],
        1,
    ),
    "seniority_title": (
        [
            "senior",
            "principal",
            "lead",
        ],
        1,
    ),
}

NEGATIVE_KEYWORDS: Dict[str, int] = {
    "sales": 3,
    "marketing": 3,
    "recruiter": 3,
    "customer support": 3,
    "account manager": 3,
    "business development": 3,
    "teacher": 3,
    "lecturer": 3,
    "qa tester": 3,
    "sap": 3,
    "service desk": 3,
    "electrical engineer": 5,
    "mechanical engineer": 5,
}

def normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

def word_in(text: str, phrase: str) -> bool:
    """Case-insensitive whole-word/phrase match using boundaries."""
    if not phrase:
        return False
    pattern = r"\b" + re.escape(phrase.lower()) + r"\b"
    return re.search(pattern, text) is not None

def extract_first_qualification_point(description: str) -> str:
    """Extract the first bullet point or requirement from qualifications section."""
    if not description:
        return ""
    
    desc_lower = description.lower()
    
    # Common section headers for qualifications
    qual_headers = [
        "qualifications",
        "requirements", 
        "required qualifications",
        "minimum qualifications",
        "essential requirements",
        "what you need",
        "you have",
        "ideal candidate",
        "we're looking for"
    ]
    
    # Find qualifications section
    qual_start = -1
    for header in qual_headers:
        pattern = r"\b" + re.escape(header) + r"\b"
        match = re.search(pattern, desc_lower)
        if match:
            qual_start = match.start()
            break
    
    if qual_start == -1:
        return ""
    
    # Extract text after qualifications header
    qual_text = description[qual_start:]
    
    # Find first bullet point or numbered item
    patterns = [
        r"[•*-]\s*([^\n\r•*-]+)",  # bullet points
        r"\d+\.\s*([^\n\r\d]+)",   # numbered items
        r"\n\s*([A-Z][^\n\r]+)",   # first capitalized line after header
    ]
    
    for pattern in patterns:
        match = re.search(pattern, qual_text, re.MULTILINE | re.IGNORECASE)
        if match:
            first_point = match.group(1).strip()
            # Clean up common artifacts
            first_point = re.sub(r'\s+', ' ', first_point)
            return first_point[:200]  # Limit length
    
    return ""

def score_text(title: str, description: str) -> Tuple[int, List[str], List[str]]:
    title_n = normalize_text(title)
    desc_n = normalize_text(description)
    
    # Extract first qualification point for enhanced scoring
    first_qual = normalize_text(extract_first_qualification_point(description))
    
    positives: List[str] = []
    negatives: List[str] = []
    score = 0.0

    # Positive keywords with title multiplier and qualifications boost
    for group, (terms, weight) in POSITIVE_KEYWORDS.items():
        for term in terms:
            matched = False
            in_title = word_in(title_n, term)
            in_desc = word_in(desc_n, term)
            in_first_qual = word_in(first_qual, term) if first_qual else False
            
            if in_title or in_desc or in_first_qual:
                matched = True
                add = float(weight)
                
                # Apply multipliers
                if in_title:
                    add *= 1.5  # title emphasis
                
                # Enhanced scoring for degree_role and core_domain in first qualification
                if in_first_qual and group in ["degree_role", "core_domain"]:
                    add *= 3  # 3x multiplier for key qualifications
                
                score += add
                positives.append(term)
                
    positives = sorted(set(positives))

    # Negative keywords with harsher penalties for title and first qualification
    for term, weight in NEGATIVE_KEYWORDS.items():
        in_title = word_in(title_n, term)
        in_desc = word_in(desc_n, term)
        in_first_qual = word_in(first_qual, term) if first_qual else False
        
        if in_title or in_desc or in_first_qual:
            penalty = float(weight)
            
            if in_title:
                penalty += 2  # title penalty boost
            
            # Enhanced penalty for negative keywords in first qualification
            if in_first_qual:
                penalty *= 5  # 5x multiplier for qualification red flags
            
            score -= penalty
            negatives.append(term)
            
    negatives = sorted(set(negatives))

    return int(round(score)), positives, negatives

def has_domain_relevance(title: str, description: str) -> bool:
    """Quick domain filter for broad search terms to avoid irrelevant jobs."""
    text = normalize_text(f"{title} {description}")
    
    # Must contain at least one domain indicator
    domain_keywords = [
        # Core sciences
        "materials", "chemistry", "physics", "computational", "quantum", 
        "molecular", "simulation", "scientific", "research",
        # Applications  
        "drug", "pharma", "battery", "catalyst", "polymer", "semiconductor",
        "protein", "crystal", "nano", "bio", "chemical", "physical",
        # Methods/tools
        "dft", "md", "monte carlo", "ab initio", "vasp", "gaussian", 
        "machine learning", "deep learning", "ai", "optimization",
        # Infrastructure
        "hpc", "cuda", "pytorch", "tensorflow", "rdkit", "openmm"
    ]
    
    return any(word_in(text, keyword) for keyword in domain_keywords)

def is_relevant(score: int, positives: List[str]) -> bool:
    # guardrail: must include at least one degree/role OR core domain signal
    has_core = any(
        term in positives
        for term in POSITIVE_KEYWORDS["core_domain"][0]
    ) or any(term in positives for term in POSITIVE_KEYWORDS["degree_role"][0])
    return has_core and score >= RELEVANCE_THRESHOLD

def adaptive_delay(site: str, consecutive_errors: int = 0) -> float:
    """Calculate delay with exponential backoff for repeated errors."""
    base_min, base_max = SITE_DELAYS.get(site, (5, 15))
    
    # Exponential backoff for consecutive errors (caps at 4x multiplier)
    error_multiplier = min(2 ** consecutive_errors, 4)
    
    adjusted_min = base_min * error_multiplier
    adjusted_max = base_max * error_multiplier
    
    return random.uniform(adjusted_min, adjusted_max)

def is_rate_limited_error(error_msg: str) -> bool:
    """Detect if an error indicates rate limiting."""
    error_lower = str(error_msg).lower()
    return any(indicator in error_lower for indicator in RATE_LIMIT_INDICATORS)

# ------------------------------
# Checkpoint Management System
# ------------------------------

class ScrapingCheckpoint:
    """Manages checkpointing and resume functionality for job scraping."""
    
    def __init__(self, checkpoint_dir: str = CHECKPOINT_DIR):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(checkpoint_dir, "progress.json")
        self.data_file = os.path.join(checkpoint_dir, "scraped_data.json")
        self.failed_file = os.path.join(checkpoint_dir, "failed_combinations.json")
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Thread lock for concurrent access
        self.lock = threading.Lock()
        
    def save_checkpoint(self, completed: set, failed: dict, current_data: list):
        """Save current progress to checkpoint files."""
        with self.lock:
            try:
                # Save progress state
                progress_data = {
                    "timestamp": datetime.now().isoformat(),
                    "completed_combinations": list(completed),
                    "total_jobs_found": len(current_data),
                    "scraping_session_id": getattr(self, 'session_id', str(int(time.time())))
                }
                
                with open(self.checkpoint_file, 'w') as f:
                    json.dump(progress_data, f, indent=2)
                
                # Save failed combinations with timestamps
                with open(self.failed_file, 'w') as f:
                    json.dump(failed, f, indent=2)
                
                # Save scraped data (as list of dicts for JSON compatibility)
                data_to_save = []
                for job_dict in current_data:
                    # Convert pandas Series to dict if needed
                    if hasattr(job_dict, 'to_dict'):
                        data_to_save.append(job_dict.to_dict())
                    else:
                        data_to_save.append(job_dict)
                
                with open(self.data_file, 'w') as f:
                    json.dump(data_to_save, f, indent=2)
                    
                print(f"  💾 Checkpoint saved: {len(completed)} completed, {len(current_data)} jobs")
                
            except Exception as e:
                print(f"  ⚠️ Failed to save checkpoint: {e}")
    
    def load_checkpoint(self):
        """Load previous progress from checkpoint files."""
        completed = set()
        failed = {}
        existing_data = []
        
        try:
            # Load progress
            if os.path.exists(self.checkpoint_file):
                with open(self.checkpoint_file, 'r') as f:
                    progress = json.load(f)
                    completed = set(progress.get("completed_combinations", []))
                    self.session_id = progress.get("scraping_session_id", str(int(time.time())))
                    print(f"  📂 Found checkpoint with {len(completed)} completed combinations")
            
            # Load failed combinations
            if os.path.exists(self.failed_file):
                with open(self.failed_file, 'r') as f:
                    failed = json.load(f)
                    print(f"  📂 Found {len(failed)} failed combinations")
            
            # Load existing data
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    existing_data = json.load(f)
                    print(f"  📂 Found {len(existing_data)} existing jobs")
                    
        except Exception as e:
            print(f"  ⚠️ Failed to load checkpoint: {e}")
            
        return completed, failed, existing_data
    
    def should_retry_failed(self, combination_key: str, failed_dict: dict) -> bool:
        """Check if enough time has passed to retry a failed combination."""
        if combination_key not in failed_dict:
            return True
            
        last_attempt = failed_dict[combination_key].get("last_attempt_time", 0)
        return time.time() - last_attempt > RETRY_FAILED_AFTER
    
    def mark_failed(self, combination_key: str, error_msg: str, failed_dict: dict):
        """Mark a combination as failed with error details."""
        failed_dict[combination_key] = {
            "error": str(error_msg),
            "last_attempt_time": time.time(),
            "attempt_count": failed_dict.get(combination_key, {}).get("attempt_count", 0) + 1,
            "is_rate_limited": is_rate_limited_error(error_msg)
        }
    
    def get_combination_key(self, search_term: str, country: str, site: str) -> str:
        """Generate unique key for combination."""
        return f"{search_term}|{country}|{site}"
    
    def cleanup_old_checkpoints(self, keep_backups: int = 3):
        """Remove old checkpoint files, keeping only recent backups."""
        try:
            # This could be enhanced to rotate checkpoint files
            pass
        except Exception as e:
            print(f"  ⚠️ Checkpoint cleanup failed: {e}")

# ------------------------------
# PART 1: SCRAPING FUNCTIONS (Independent)
# ------------------------------

def scrape_single_combination(search_term: str, country: str, site: str, seen_urls: set, consecutive_errors: dict) -> Tuple[list, int, str]:
    """Scrape a single search term/country/site combination."""
    jobs_found = []
    error_msg = ""
    duplicates_found = 0
    
    try:
        country_indeed = get_country_indeed(country)
        
        kwargs = dict(
            site_name=site,
            search_term=search_term,
            results_wanted=RESULTS_PER_COMBO,
            job_type="fulltime",
            hours_old=HOURS_OLD,
            country_indeed=country_indeed,
            verbose=0,  # Reduced verbosity for parallel execution
            description_format=DESCRIPTION_FORMAT,
        )
        
        if site == "indeed":
            kwargs["location"] = None
            kwargs["google_search_term"] = None
        elif site == "glassdoor":
            kwargs["location"] = country
            kwargs["google_search_term"] = None
        elif site == "google":
            kwargs["location"] = None
            kwargs["google_search_term"] = f"{search_term} jobs in {country} in the last 2 months"
        elif site == "linkedin":
            kwargs["location"] = country
            kwargs["distance"] = 100
            kwargs["linkedin_fetch_description"] = True
            kwargs["google_search_term"] = None
        
        jobs_df = scrape_jobs(**kwargs)
        consecutive_errors[site] = 0  # Reset on success
        
        if jobs_df is not None and not jobs_df.empty:
            # Incremental deduplication
            new_jobs = jobs_df[~jobs_df["job_url"].isin(seen_urls)]
            duplicates_found = len(jobs_df) - len(new_jobs)
            
            # Domain filtering for broad search terms
            if search_term in BROAD_TERMS_NEEDING_FILTER and len(new_jobs) > 0:
                domain_mask = new_jobs.apply(
                    lambda row: has_domain_relevance(
                        row.get("title", ""), 
                        row.get("description", "")
                    ), axis=1
                )
                new_jobs = new_jobs[domain_mask]
            
            if len(new_jobs) > 0:
                new_jobs["search_term_used"] = search_term
                new_jobs["search_country"] = country
                new_jobs["source_site"] = site
                jobs_found = new_jobs.to_dict('records')
                seen_urls.update(new_jobs["job_url"])
        
    except Exception as e:
        error_msg = str(e)
        consecutive_errors[site] = consecutive_errors.get(site, 0) + 1
    
    return jobs_found, duplicates_found, error_msg

def scrape_raw_jobs_parallel(resume: bool = False) -> pd.DataFrame:
    """Scrape jobs with parallel execution and checkpoint/resume functionality."""
    
    # Initialize checkpoint system
    checkpoint = ScrapingCheckpoint()
    
    # Load existing progress if resuming
    completed_combinations = set()
    failed_combinations = {}
    all_jobs = []
    seen_urls = set()
    
    if resume:
        print("🔄 Resuming from checkpoint...")
        completed_combinations, failed_combinations, existing_data = checkpoint.load_checkpoint()
        
        # Convert existing data back to jobs list and rebuild seen_urls
        for job_dict in existing_data:
            all_jobs.append(job_dict)
            if 'job_url' in job_dict:
                seen_urls.add(job_dict['job_url'])
        
        print(f"  📊 Loaded {len(all_jobs)} existing jobs, {len(seen_urls)} URLs tracked")
    
    # Build list of combinations to process
    all_combinations = []
    for search_term in SEARCH_TERMS:
        for country in COUNTRIES:
            for site in SITES:
                combo_key = checkpoint.get_combination_key(search_term, country, site)
                
                # Skip if already completed (unless retrying failed)
                if combo_key in completed_combinations:
                    continue
                    
                # Skip failed combinations unless enough time has passed
                if combo_key in failed_combinations and not checkpoint.should_retry_failed(combo_key, failed_combinations):
                    continue
                
                all_combinations.append((search_term, country, site, combo_key))
    
    total_combinations = len(all_combinations)
    print(f"🚀 Starting parallel scrape: {total_combinations} combinations to process")
    print(f"  📊 Checkpoint: {len(completed_combinations)} completed, {len(failed_combinations)} previously failed")
    print(f"  ⚡ Parallel sites: {MAX_PARALLEL_SITES}")
    print("-" * 60)
    
    consecutive_errors = {}
    combination_count = 0
    
    # Group combinations by search term and country for organized processing
    for search_term in SEARCH_TERMS:
        for country in COUNTRIES:
            current_combinations = [
                (st, c, site, key) for st, c, site, key in all_combinations 
                if st == search_term and c == country
            ]
            
            if not current_combinations:
                continue
                
            print(f"\n📍 Processing: '{search_term}' in {country}")
            
            # Process sites for this search term/country in parallel
            with ThreadPoolExecutor(max_workers=MAX_PARALLEL_SITES) as executor:
                # Submit all site tasks for this combination
                future_to_combo = {}
                for st, c, site, combo_key in current_combinations:
                    future = executor.submit(
                        scrape_single_combination, 
                        st, c, site, seen_urls, consecutive_errors
                    )
                    future_to_combo[future] = (st, c, site, combo_key)
                
                # Process results as they complete
                for future in as_completed(future_to_combo):
                    st, c, site, combo_key = future_to_combo[future]
                    combination_count += 1
                    
                    try:
                        jobs_found, duplicates_found, error_msg = future.result()
                        
                        if error_msg:
                            print(f"  ⚠️  {site:10s} error: {error_msg[:50]}...")
                            checkpoint.mark_failed(combo_key, error_msg, failed_combinations)
                            
                            # Apply adaptive delay for rate-limited errors
                            if is_rate_limited_error(error_msg):
                                error_count = consecutive_errors.get(site, 0)
                                sleep_time = adaptive_delay(site, error_count)
                                print(f"  🕐 Rate limit detected, waiting {sleep_time:.1f}s...")
                                time.sleep(sleep_time)
                        else:
                            # Success
                            completed_combinations.add(combo_key)
                            all_jobs.extend(jobs_found)
                            
                            if jobs_found:
                                print(f"  ✅ {site:10s} → {len(jobs_found)} new jobs ({duplicates_found} duplicates)")
                            else:
                                print(f"  🔄 {site:10s} → 0 new jobs ({duplicates_found} all duplicates)")
                    
                    except Exception as e:
                        print(f"  💥 {site:10s} unexpected error: {e}")
                        checkpoint.mark_failed(combo_key, str(e), failed_combinations)
                    
                    # Save checkpoint periodically
                    if combination_count % CHECKPOINT_INTERVAL == 0:
                        checkpoint.save_checkpoint(completed_combinations, failed_combinations, all_jobs)
                    
                    # Apply site-specific delay after each combination
                    error_count = consecutive_errors.get(site, 0)
                    sleep_time = adaptive_delay(site, error_count)
                    time.sleep(sleep_time)
    
    # Final checkpoint save
    checkpoint.save_checkpoint(completed_combinations, failed_combinations, all_jobs)
    
    if not all_jobs:
        print("❌ No jobs found across all searches")
        return pd.DataFrame()
    
    # Convert to DataFrame and deduplicate
    combined = pd.DataFrame(all_jobs)
    pre_final_dedup = len(combined)
    combined = combined.drop_duplicates(subset=["job_url"], keep="first")
    final_dedup_removed = pre_final_dedup - len(combined)
    
    print(f"\n📊 Parallel Scraping Summary:")
    print(f"  • Total API calls made: {combination_count}")
    print(f"  • Completed combinations: {len(completed_combinations)}")
    print(f"  • Failed combinations: {len(failed_combinations)}")
    print(f"  • Final deduplication removed: {final_dedup_removed}")
    print(f"  • Total unique jobs found: {len(combined)}")
    
    return combined

# Legacy function for backwards compatibility
def scrape_raw_jobs() -> pd.DataFrame:
    """Legacy sequential scraping function."""
    return scrape_raw_jobs_parallel(resume=False)

# ------------------------------
# PART 2: FILTERING/SCORING FUNCTIONS (Independent)
# ------------------------------

def filter_and_score_jobs(jobs_df: pd.DataFrame, save_output: bool = True, output_file: str = "ai_science_jobs_europe_phd.csv") -> pd.DataFrame:
    """Apply filtering and scoring to raw job data."""
    if jobs_df.empty:
        print("No jobs to filter.")
        return jobs_df
    
    print(f"\nStarting filtering and scoring of {len(jobs_df)} jobs...")
    
    filtered_jobs = []
    filtered_count = 0
    
    for _, job in jobs_df.iterrows():
        job_dict = job.to_dict()
        search_term = job_dict.get("search_term", "")
        
        # Apply domain filtering for broad terms
        if search_term in BROAD_TERMS_NEEDING_FILTER:
            if not has_domain_relevance(job_dict.get("title", ""), job_dict.get("description", "")):
                filtered_count += 1
                continue
        
        # Calculate relevance score with enhanced qualifications parsing
        score, positives, negatives = score_text(
            job_dict.get("title", ""), 
            job_dict.get("description", "")
        )
        
        # Extract first qualification for debugging
        first_qual = extract_first_qualification_point(job_dict.get("description", ""))
        
        job_dict["relevance_score"] = score
        job_dict["positive_keywords"] = positives
        job_dict["negative_keywords"] = negatives
        job_dict["first_qualification"] = first_qual  # Store for analysis
        
        # Apply relevance check (score + guardrail)
        if is_relevant(score, positives):
            filtered_jobs.append(job_dict)
    
    if not filtered_jobs:
        print("No jobs passed filtering.")
        return pd.DataFrame()
    
    # Convert to DataFrame
    filtered_df = pd.DataFrame(filtered_jobs)
    
    # Sort by relevance score (descending) and date (newest first)
    filtered_df = filtered_df.sort_values(
        ["relevance_score", "date_posted"], 
        ascending=[False, False]
    ).reset_index(drop=True)
    
    print(f"Filtering results: {len(jobs_df)} → {len(filtered_df)} jobs ({filtered_count} filtered out)")
    
    if save_output:
        # Save to CSV
        filtered_df.to_csv(output_file, index=False, quoting=csv.QUOTE_ALL)
        print(f"Filtered jobs saved to: {output_file}")
    
    return filtered_df

def print_filtering_summary(jobs_df: pd.DataFrame):
    """Print summary statistics for filtered jobs."""
    if jobs_df.empty:
        return
    
    print(f"\n" + "=" * 60)
    print("FILTERING & SCORING SUMMARY")
    print("=" * 60)
    print(f"Total jobs after filtering: {len(jobs_df)}")
    
    print(f"\nTop relevance scores:")
    print(jobs_df["relevance_score"].value_counts().sort_index(ascending=False).head(10))
    
    print(f"\nJobs by site:")
    print(jobs_df["site"].value_counts())
    
    print(f"\nJobs by country:")
    print(jobs_df["search_country"].value_counts())
    
    if len(jobs_df) > 0:
        print(f"\nTop 5 most relevant jobs:")
        for i, (_, job) in enumerate(jobs_df.head(5).iterrows()):
            positive_kw = job.get('positive_keywords', [])[:3]  # Show top 3 keywords
            print(f"{i+1}. [{job['relevance_score']}] {job['title']} at {job.get('company', 'Unknown')}")
            print(f"    Location: {job['search_country']}, Site: {job['site']}")
            print(f"    Keywords: {', '.join(positive_kw)}")
            if job.get('first_qualification'):
                print(f"    First Qualification: {job['first_qualification'][:100]}...")
            print()

# ------------------------------
# MAIN EXECUTION MODES
# ------------------------------

def main_scrape_only(resume: bool = False, parallel: bool = True):
    """Mode 1: Scrape jobs only (no filtering)."""
    mode_name = "PARALLEL SCRAPING" if parallel else "SEQUENTIAL SCRAPING" 
    resume_text = " - RESUME MODE" if resume else ""
    
    print("=" * 60)
    print(f"JOB SCRAPER - {mode_name}{resume_text}")
    print("=" * 60)
    
    # Choose scraping method
    if parallel:
        jobs_df = scrape_raw_jobs_parallel(resume=resume)
    else:
        jobs_df = scrape_raw_jobs()
    
    if not jobs_df.empty:
        # Save raw data with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_output_file = f"raw_jobs_scraped_{timestamp}.csv" if not resume else "raw_jobs_scraped.csv"
        jobs_df.to_csv(raw_output_file, index=False, quoting=csv.QUOTE_ALL)
        print(f"\nRaw jobs saved to: {raw_output_file}")
        
        print(f"\nRaw scraping summary:")
        print(f"Total jobs: {len(jobs_df)}")
        if 'source_site' in jobs_df.columns:
            print(f"Jobs by site: {jobs_df['source_site'].value_counts().to_dict()}")
        if 'search_country' in jobs_df.columns:
            print(f"Jobs by country: {jobs_df['search_country'].value_counts().to_dict()}")

def main_filter_only(input_file: str = "raw_jobs_scraped.csv"):
    """Mode 2: Filter and score existing scraped data."""
    print("=" * 60)
    print("JOB SCRAPER - FILTERING ONLY MODE")
    print("=" * 60)
    
    try:
        # Load raw data
        jobs_df = pd.read_csv(input_file)
        print(f"Loaded {len(jobs_df)} jobs from {input_file}")
        
        # Filter and score
        filtered_df = filter_and_score_jobs(jobs_df)
        
        # Print summary
        print_filtering_summary(filtered_df)
        
    except FileNotFoundError:
        print(f"Error: {input_file} not found. Run scrape mode first.")
    except Exception as e:
        print(f"Error loading data: {str(e)}")

def main_full_pipeline(resume: bool = False, parallel: bool = True):
    """Mode 3: Full pipeline (scrape + filter)."""
    mode_name = "PARALLEL" if parallel else "SEQUENTIAL"
    resume_text = " - RESUME MODE" if resume else ""
    
    print("🔬 PhD-relevant jobs scraper (computational chemistry/physics) across Europe")
    print(f"🎯 {mode_name} execution with weighted relevance scoring{resume_text}")
    print()
    
    # Choose scraping method
    if parallel:
        jobs_df = scrape_raw_jobs_parallel(resume=resume)
    else:
        jobs_df = scrape_raw_jobs()
    
    if jobs_df.empty:
        print("No jobs to process.")
        return
    
    # Save raw data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_output_file = f"raw_jobs_scraped_{timestamp}.csv"
    jobs_df.to_csv(raw_output_file, index=False, quoting=csv.QUOTE_ALL)
    print(f"\nRaw jobs saved to: {raw_output_file}")
    
    # Filter and score
    filtered_df = filter_and_score_jobs(jobs_df)
    
    # Print summary
    print_filtering_summary(filtered_df)

def print_usage():
    """Print usage information."""
    print("Enhanced Job Scraper with Checkpoint/Resume Support")
    print("=" * 50)
    print("Usage: python job_scraper_split.py [MODE] [OPTIONS]")
    print()
    print("MODES:")
    print("  scrape       - Scrape jobs only (no filtering)")
    print("  filter       - Filter and score existing scraped data") 
    print("  full         - Full pipeline (scrape + filter)")
    print("  resume       - Resume from last checkpoint")
    print("  retry-failed - Retry previously failed combinations")
    print("  status       - Show checkpoint status")
    print()
    print("OPTIONS:")
    print("  --sequential - Use sequential scraping (default: parallel)")
    print("  --input FILE - Specify input file for filter mode")
    print("  --parallel N - Number of parallel sites (default: 4)")
    print()
    print("EXAMPLES:")
    print("  python job_scraper_split.py scrape")
    print("  python job_scraper_split.py resume")
    print("  python job_scraper_split.py filter --input raw_jobs_scraped.csv")
    print("  python job_scraper_split.py scrape --sequential")

def show_checkpoint_status():
    """Show current checkpoint status."""
    checkpoint = ScrapingCheckpoint()
    completed, failed, existing_data = checkpoint.load_checkpoint()
    
    total_combinations = len(SEARCH_TERMS) * len(COUNTRIES) * len(SITES)
    remaining = total_combinations - len(completed)
    
    print("📊 CHECKPOINT STATUS")
    print("=" * 40)
    print(f"Total combinations: {total_combinations}")
    print(f"Completed: {len(completed)}")
    print(f"Failed: {len(failed)}")
    print(f"Remaining: {remaining}")
    print(f"Jobs scraped: {len(existing_data)}")
    print(f"Progress: {len(completed)/total_combinations*100:.1f}%")
    
    if failed:
        print(f"\nRecent failures:")
        for combo, details in list(failed.items())[-5:]:
            error = details.get('error', 'Unknown')[:50]
            attempts = details.get('attempt_count', 0)
            print(f"  {combo}: {error}... (attempts: {attempts})")

if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    if len(sys.argv) < 2:
        main_full_pipeline(parallel=True)
        sys.exit()
    
    mode = sys.argv[1].lower()
    args = sys.argv[2:]
    
    # Parse options
    sequential = "--sequential" in args
    parallel = not sequential
    parallel_count = MAX_PARALLEL_SITES
    input_file = "raw_jobs_scraped.csv"
    
    # Extract specific options
    if "--input" in args:
        idx = args.index("--input")
        if idx + 1 < len(args):
            input_file = args[idx + 1]
    
    if "--parallel" in args:
        idx = args.index("--parallel")
        if idx + 1 < len(args):
            try:
                parallel_count = int(args[idx + 1])
                # Update global config temporarily
                globals()['MAX_PARALLEL_SITES'] = parallel_count
            except ValueError:
                print("Invalid parallel count specified")
                sys.exit(1)
    
    # Execute based on mode
    try:
        if mode == "scrape":
            main_scrape_only(resume=False, parallel=parallel)
        elif mode == "filter":
            main_filter_only(input_file)
        elif mode == "full":
            main_full_pipeline(resume=False, parallel=parallel)
        elif mode == "resume":
            main_full_pipeline(resume=True, parallel=parallel)
        elif mode == "resume-scrape":
            main_scrape_only(resume=True, parallel=parallel)
        elif mode == "retry-failed":
            # Clear the failed combinations file to retry all
            checkpoint = ScrapingCheckpoint()
            if os.path.exists(checkpoint.failed_file):
                os.remove(checkpoint.failed_file)
                print("🔄 Cleared failed combinations - they will be retried")
            main_full_pipeline(resume=True, parallel=parallel)
        elif mode == "status":
            show_checkpoint_status()
        elif mode in ["help", "--help", "-h"]:
            print_usage()
        else:
            print(f"Unknown mode: {mode}")
            print_usage()
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Scraping interrupted by user")
        print("💾 Progress has been saved to checkpoint files")
        print("🔄 Run with 'resume' mode to continue from where you left off")
        sys.exit(0)
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        print("💾 Check checkpoint files for any saved progress")
        sys.exit(1)
