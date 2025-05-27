import streamlit as st
st.set_page_config(layout="wide", page_title="NCEA LLM Grading App")
import pandas as pd
import json
import re
import fitz  # PyMuPDF
from bs4 import BeautifulSoup
from openai import OpenAI
from typing import Optional, List, Dict, Any
from pathlib import Path
from collections import Counter
import zipfile
import os
import tempfile
import logging
import time
from docx import Document

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Constants and Configuration ---
GRADE_ORDER = ['N1', 'N2', 'A3', 'A4', 'M5', 'M6', 'E7', 'E8']

def build_grading_prompt(rubric_data: dict, student_text: str) -> str:
    """Build a prompt for grading student work based on the rubric, requesting per-criterion judgments."""
    as_code = rubric_data.get('as_code', 'Unknown')
    as_title = rubric_data.get('as_title', 'Unknown Title')
    criteria_section = ""

    if 'gradingCriteria' in rubric_data:
        for level_info in rubric_data['gradingCriteria']:
            level_name = level_info.get('levelName', '')
            sublevels = level_info.get('sublevels', [])
            main_req = level_info.get('mainRequirement', '')
            criteria_involves = level_info.get('criteriaInvolves', [])
            evidence_focus_points = level_info.get('evidenceFocusPoints', [])
            example_clarification = level_info.get('exampleClarification', '')

            criteria_section += f"\n{level_name} ({', '.join(sublevels)}):\n"
            criteria_section += f"Main Requirement: {main_req}\n"

            criteria_section += "Criteria Involves:\n"
            for criterion in criteria_involves:
                criteria_section += f"    - {criterion}\n"

            criteria_section += "Evidence Focus Points:\n"
            for point in evidence_focus_points:
                criteria_section += f"    - {point}\n"

            criteria_section += f"Example Clarification: {example_clarification}\n"

    prompt = f"""You are an expert NCEA (New Zealand Certificate of Educational Achievement) Assessment Assistant.
Your task is to grade a student portfolio for {as_title} (AS{as_code}) based on the provided rubric.

### Rubric Criteria:
{criteria_section}

### Grading Instructions:
1. For each criterion, provide a boolean (true/false) indicating if it is met, grouped by level (e.g., {{"A": [true, false, ...], "M": [true, ...], ...}}).
2. Identify which criteria the student has met and which they have NOT met.
3. Provide brief evidence from the student's work for each criterion.
4. Determine the appropriate grade based on the highest level where ALL criteria are met.
5. Ensure that students meet ALL criteria from lower grades to achieve higher grades.
6. The final grade should be the highest sublevel where ALL criteria are fully met.

### Response Format:
Respond with a JSON object in the following structure:
{{
  "grade": "A4",
  "justification": "Clear explanation of why this grade was given",
  "failed_criteria": ["List any criteria the student failed to meet"],
  "confidence_flags": ["List any areas where grading was difficult/uncertain"],
  "per_bullet": {{"A": [true, false, ...], "M": [true, ...], ...}}
}}

### Example Response:
{{
  "grade": "M5",
  "justification": "The student met all criteria for Merit but failed one criterion for Excellence.",
  "failed_criteria": ["Criterion 3: Evidence of critical thinking"],
  "confidence_flags": ["Limited evidence for Criterion 2"],
  "per_bullet": {{"A": [true, true, true], "M": [true, true], "E": [false]}}
}}

### Student Work:
---
{student_text}
---

Do not include any text outside the JSON object."""
    return prompt

def validate_pdf_path(pdf_path: Path) -> bool:
    """Validate if the provided PDF path exists and is a file."""
    if not pdf_path or not pdf_path.exists():
        st.error(f"PDF path invalid or file does not exist: {pdf_path}")
        return False
    return True

def extract_text_from_pdf(pdf_path: Path) -> Optional[str]:
    """Extract text from a PDF file using PyMuPDF."""
    if not validate_pdf_path(pdf_path):
        return None

    try:
        text = ""
        with fitz.open(pdf_path) as doc:
            if not doc or doc.page_count == 0:
                st.warning(f"PDF document {pdf_path.name} appears to be empty or invalid")
                return None

            for page_idx in range(doc.page_count):
                page = doc[page_idx]
                raw_text = page.get_text("text")
                html_text = page.get_text("html")

                # Clean HTML text
                cleaned_html_text = ""
                if html_text:
                    try:
                        soup = BeautifulSoup(html_text, 'html.parser')
                        cleaned_html_text = soup.get_text(separator=' ', strip=True)
                    except Exception as e:
                        logging.warning(f"Error cleaning HTML text on page {page_idx}: {e}")

                # Choose best extraction
                page_text = raw_text if raw_text.strip() else cleaned_html_text

                # Check for images if minimal text
                if len(page_text.strip()) < 100:
                    image_list = page.get_images(full=True)
                    if image_list and not page_text.strip():
                        page_text += f"[Note: Page {page_idx+1} contains {len(image_list)} image(s) that may have text not extracted] "

                text += page_text + "\n\n"

        final_text = text.strip()
        if not final_text:
            st.warning(f"Could not extract any text from {pdf_path.name}. The PDF may be scanned images or have security restrictions.")
            return None

        return final_text

    except Exception as e:
        st.error(f"Error extracting text from {pdf_path.name}: {str(e)}")
        logging.error(f"Error extracting text from {pdf_path}: {e}")
        return None

def extract_text_from_docx(docx_path: Path) -> Optional[str]:
    """Extract text from a DOCX file using python-docx with smart content prioritization."""
    try:
        doc = Document(docx_path)
        
        # Extract text from paragraphs with priority on content
        text_content = []
        
        # Prioritize headings and main content
        for paragraph in doc.paragraphs:
            text = paragraph.text.strip()
            if text:
                # Skip very short lines that might be formatting artifacts
                if len(text) > 3:
                    text_content.append(text)
        
        # Extract text from tables (often contains important data)
        for table in doc.tables:
            for row in table.rows:
                row_text = []
                for cell in row.cells:
                    cell_text = cell.text.strip()
                    if cell_text and len(cell_text) > 1:
                        row_text.append(cell_text)
                if row_text:
                    text_content.append(" | ".join(row_text))
        
        # Join content with appropriate spacing
        final_text = "\n".join(text_content)
        
        # Clean up excessive whitespace
        import re
        final_text = re.sub(r'\n\s*\n', '\n\n', final_text)  # Remove empty lines
        final_text = re.sub(r' +', ' ', final_text)  # Remove multiple spaces
        
        if not final_text.strip():
            st.warning(f"Could not extract any meaningful text from {docx_path.name}. The document may be empty or contain only formatting.")
            return None
        
        # Log extraction info for debugging (only log once per session)
        if f"docx_extracted_{docx_path.name}" not in st.session_state:
            st.info(f"📄 Extracted {len(final_text)} characters from {docx_path.name}")
            st.session_state[f"docx_extracted_{docx_path.name}"] = True
        
        return final_text
        
    except Exception as e:
        st.error(f"Error extracting text from {docx_path.name}: {str(e)}")
        logging.error(f"Error extracting text from {docx_path}: {e}")
        return None

def smart_truncate_text(text: str, max_chars: int, file_name: str = "") -> str:
    """Intelligently truncate text while preserving important content."""
    if len(text) <= max_chars:
        return text
    
    # Split into sections/paragraphs
    paragraphs = text.split('\n\n')
    
    # Prioritize content (look for key indicators)
    important_keywords = [
        'achievement', 'standard', 'criteria', 'evidence', 'conclusion', 
        'method', 'result', 'analysis', 'evaluation', 'recommendation',
        'introduction', 'aim', 'hypothesis', 'discussion', 'summary'
    ]
    
    # Score paragraphs by importance
    scored_paragraphs = []
    for i, paragraph in enumerate(paragraphs):
        score = 0
        para_lower = paragraph.lower()
        
        # Higher score for paragraphs with important keywords
        for keyword in important_keywords:
            score += para_lower.count(keyword) * 10
        
        # Prefer paragraphs near the beginning and end
        if i < len(paragraphs) * 0.3:  # First 30%
            score += 20
        elif i > len(paragraphs) * 0.7:  # Last 30%
            score += 10
        
        # Prefer longer paragraphs (more substantial content)
        if len(paragraph) > 100:
            score += 5
        
        scored_paragraphs.append((score, paragraph, i))
    
    # Sort by score (highest first)
    scored_paragraphs.sort(key=lambda x: x[0], reverse=True)
    
    # Build truncated text
    result_text = ""
    used_chars = 0
    
    for score, paragraph, original_index in scored_paragraphs:
        # Add paragraph if it fits
        if used_chars + len(paragraph) + 4 <= max_chars:  # +4 for spacing
            if result_text:
                result_text += "\n\n"
            result_text += paragraph
            used_chars = len(result_text)
        elif used_chars < max_chars * 0.8:  # Still have significant space
            # Try to fit a truncated version of this paragraph
            remaining_space = max_chars - used_chars - 50  # Leave space for truncation notice
            if remaining_space > 100:  # Only if meaningful space left
                if result_text:
                    result_text += "\n\n"
                result_text += paragraph[:remaining_space] + "..."
                break
        else:
            break
    
    if len(result_text) < max_chars * 0.5:
        # Fallback: simple truncation if smart truncation didn't work well
        result_text = text[:max_chars]
    
    # Add truncation notice
    if len(text) > len(result_text):
        result_text += f"\n\n[Content truncated from {len(text)} to {len(result_text)} characters for processing]"
    
    return result_text

def extract_text_from_file(file_path: Path) -> Optional[str]:
    """Extract text from either PDF or DOCX files."""
    file_extension = file_path.suffix.lower()
    
    if file_extension == '.pdf':
        return extract_text_from_pdf(file_path)
    elif file_extension == '.docx':
        return extract_text_from_docx(file_path)
    else:
        st.error(f"Unsupported file type: {file_extension}")
        return None

def explain_failed_criteria_with_llm(
    openrouter_client: OpenAI,
    api_key: str, 
    failed_criteria: List[str], 
    student_text: str, 
    model_name: str, 
    max_tokens: int
) -> List[Dict[str, str]]:
    """Get in-depth explanations for failed criteria using an LLM."""
    if not failed_criteria:
        return []
        
    if not api_key:
        st.error("API key missing for explain_failed_criteria_with_llm.")
        return [{"criterion": fc, "explanation": "Error: API key missing."} for fc in failed_criteria]

    try:
        # Truncate student text to avoid hitting token limits
        max_student_chars = 10000
        truncated_student_text = student_text[:max_student_chars]
        if len(student_text) > max_student_chars:
            truncated_student_text += "... [text truncated]"
        
        criteria_list_str = "\n".join([f"- {idx+1}. {criterion}" for idx, criterion in enumerate(failed_criteria)])
        
        prompt = f"""You are an educational assessment expert. Analyze why the student failed to meet the following criteria in their work:

{criteria_list_str}

For each criterion, provide:
1. A specific explanation of why the student's work didn't meet this criterion
2. Evidence from their work that supports your explanation
3. Brief advice on how the student could have met this criterion

Use language and phrasing from the original rubric where appropriate to explain why the student's work does not meet each criterion.

FORMAT YOUR RESPONSE AS A JSON ARRAY, with one object per failed criterion:
[
  {{
    "criterion": "<first criterion text>",
    "explanation": "<detailed explanation with evidence>"
  }},
  {{
    "criterion": "<second criterion text>",
    "explanation": "<detailed explanation with evidence>"
  }}
]

Here is the student's work:
---
{truncated_student_text}
---"""

        response = openrouter_client.chat.completions.create(
            model=model_name,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You analyze student work and explain why specific criteria were not met. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.18,
            max_tokens=max_tokens
        )
        
        result_json = response.choices[0].message.content
        
        try:
            parsed_json = json.loads(result_json)
            
            # Handle different response formats
            explanations = []
            if isinstance(parsed_json, list):
                explanations = parsed_json
            elif isinstance(parsed_json, dict):
                for key in ["explanations", "criteria", "results", "analysis"]:
                    if key in parsed_json and isinstance(parsed_json[key], list):
                        explanations = parsed_json[key]
                        break
            
            if explanations:
                formatted_explanations = []
                for i, expl in enumerate(explanations):
                    if isinstance(expl, dict):
                        criterion = expl.get("criterion", failed_criteria[i] if i < len(failed_criteria) else f"Criterion {i+1}")
                        explanation = expl.get("explanation", "No detailed explanation provided")
                        formatted_explanations.append({"criterion": criterion, "explanation": explanation})
                
                return formatted_explanations
            else:
                return [{"criterion": fc, "explanation": "Could not extract specific explanation from LLM response."} for fc in failed_criteria]
                
        except json.JSONDecodeError:
            return [{"criterion": fc, "explanation": "Error parsing LLM response."} for fc in failed_criteria]
            
    except Exception as e:
        st.error(f"Error in explain_failed_criteria_with_llm: {str(e)}")
        return [{"criterion": fc, "explanation": f"Error analyzing criterion: {str(e)}"} for fc in failed_criteria]

def run_overall_misconception_analysis(
    openrouter_client: OpenAI,
    insights_df_source: pd.DataFrame,
    llm_rubric_json: Dict,
    model_name: str,
    max_tokens: int,
    default_title: str = "the NCEA Standard",
    default_standard_code: str = "XXXXX"
) -> str:
    """Analyzes overall misconceptions based on aggregated failed criteria."""
    try:
        as_code_num = llm_rubric_json.get("as_code", "UNKNOWN")
        as_title_text = llm_rubric_json.get("as_title", "Unknown Title")

        if as_code_num != "UNKNOWN" and as_title_text != "Unknown Title":
            rubric_title_display = f"{as_title_text} (AS{as_code_num})"
        elif as_title_text != "Unknown Title":
            rubric_title_display = as_title_text
        elif as_code_num != "UNKNOWN":
            rubric_title_display = f"AS{as_code_num}"
        else:
            rubric_title_display = "Unknown Standard"

        st.subheader(f"Overall Misconception Analysis for {rubric_title_display}")

        all_explanations = []
        for _idx, row in insights_df_source.iterrows():
            fc_list = row.get('failed_criteria_indepth', [])

            if isinstance(fc_list, str):
                try:
                    fc_list = json.loads(fc_list)
                except json.JSONDecodeError:
                    logging.warning(f"Failed to parse failed_criteria_indepth JSON string for row {_idx}. Skipping row.")
                    continue

            if isinstance(fc_list, list):
                for item in fc_list:
                    if isinstance(item, dict) and 'explanation' in item:
                        all_explanations.append(item['explanation'])

        unique_explanations = list(set(all_explanations))

        # Enhance misconception descriptions with summaries and causes
        prompt = f"""You are an NCEA Technology teacher assistant.

Your job is to analyze common misconceptions in student portfolios and return insights in a simple, teacher-friendly format.

Input: A list of failed criterion explanations, each written by a grader after reviewing student work.

Instructions:
1. Group the explanations into 3–5 misconception themes (e.g., "Shallow Material Analysis" or "Weak Justification").
2. For each theme:
   - Provide a 3–4 sentence summary that explains:
    1. what the misconception is,
    2. why it arises,
    3. how it appears in student work,
    4. what kind of student it commonly affects.
   - List **2 concrete, actionable teaching strategies** teachers could use next lesson.
   - Use **bold** for keywords and *italics* for strategies (e.g., *Use a testing template*).
   - Keep language plain, short, and easy to read in class or at a department meeting.

Format your answer in Markdown, using clear headers and bullet points.
Do not repeat explanations. Avoid overwhelming detail.

Example:

### 🧠 Weak Material Justification
Students name the material but not why it suits the purpose. This often occurs due to a lack of understanding of material properties or insufficient examples provided during teaching.

**Teaching Strategies:**
- *Use peer review to improve justification clarity.*
- *Provide sentence frames like "I chose X because..."*
- *Use a checklist with function, durability, and user needs.*

MISCONCEPTION EXAMPLES:
{json.dumps(unique_explanations[:10], indent=2)}
"""

        try:
            response = openrouter_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are an educational analyst specializing in NCEA achievement level analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.6,  # Hardcoded temperature for insights
                max_tokens=max_tokens
            )

            return response.choices[0].message.content

        except Exception as e:
            st.error(f"Error during LLM call for Overall Misconception Analysis: {str(e)}")
            return f"### Overall Misconception Analysis\n\nError: Could not complete analysis. Details: {str(e)}"

    except Exception as e:
        st.error(f"Critical error in setting up Overall Misconception Analysis: {str(e)}")
        return f"### Overall Misconception Analysis\n\nError: Could not set up analysis. Details: {str(e)}"

def run_per_level_misconception_analysis(
    openrouter_client: OpenAI,
    insights_df_source: pd.DataFrame,
    llm_rubric_json: Dict,
    model_name: str,
    max_tokens: int
) -> str:
    """Analyzes misconceptions per NCEA level based on aggregated failed criteria."""
    try:
        as_code_num = llm_rubric_json.get("as_code", "UNKNOWN")
        as_title_text = llm_rubric_json.get("as_title", "Unknown Title")
        
        if as_code_num != "UNKNOWN" and as_title_text != "Unknown Title":
            rubric_title_display = f"{as_title_text} (AS{as_code_num})"
        else:
            rubric_title_display = f"{as_title_text} (AS{as_code_num})"
        
        st.subheader(f"Per-Level Misconception Analysis for {rubric_title_display}")

        ncea_levels = {
            'Not Achieved': ['N0', 'N1', 'N2'],
            'Achieved': ['A3', 'A4'],
            'Merit': ['M5', 'M6'],
            'Excellence': ['E7', 'E8']
        }
        
        analysis_results = []
        
        for level_name, grades in ncea_levels.items():
            level_data = insights_df_source[insights_df_source['grade'].isin(grades)]
            
            if len(level_data) == 0:
                continue
            
            level_explanations = []
            for _idx, row in level_data.iterrows():
                fc_list = row.get('failed_criteria_indepth', [])
                if isinstance(fc_list, list):
                    for item in fc_list:
                        if isinstance(item, dict) and 'explanation' in item:
                            level_explanations.append(item['explanation'])
            
            if not level_explanations:
                continue
            
            counts = Counter(level_explanations)
            common_issues = counts.most_common(5)
            
            student_count = len(level_data)
            criteria_summary = "\n".join([f"- '{item}': {count}/{student_count} students ({(count/student_count)*100:.1f}%)" 
                                          for item, count in common_issues])
            
            prompt = f"""You are an expert in NCEA assessment and learning design. Analyze student misconceptions based on their grading outcomes at the {level_name} level for the standard {as_title_text} (AS{as_code_num}).

STUDENT COUNT: {student_count}

COMMON FAILED CRITERIA:
{criteria_summary}

Write a focused and teacher-friendly analysis that includes:

1. 🔍 **Key Misconceptions**: Clearly explain the most common misunderstandings or learning gaps preventing students from meeting the criteria at this level.
2. 🧠 **Underlying Reasons**: Provide a brief explanation of *why* these issues are occurring (e.g. conceptual confusion, lack of modeling, insufficient scaffolding).
3. 🛠️ **Teaching Actions**: Suggest **concrete, practical teaching strategies** or activities to help students address these misconceptions and move toward the next grade band. Use specific verbs like “model,” “discuss,” “practice,” “revise,” etc.

Make the analysis **concise, practical, and written in plain English** suitable for busy teachers. Avoid repeating identical examples and focus on what teachers can *do next*.

Use markdown formatting with headers (`##`) and bullet points for readability. 
"""


            try:
                response = openrouter_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are an educational analyst specializing in NCEA achievement level analysis."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.6,  # Hardcoded temperature for insights
                    max_tokens=max_tokens
                )
                
                level_analysis = response.choices[0].message.content
                analysis_results.append(f"## {level_name} Level ({', '.join(grades)})\n\n{level_analysis}\n\n")
                
            except Exception as e_inner:
                st.error(f"Error during LLM call for {level_name} level analysis: {str(e_inner)}")
                analysis_results.append(f"## {level_name} Level ({', '.join(grades)})\n\nError in analysis: {str(e_inner)}\n\n")
        
        if analysis_results:
            final_analysis = f"# Per-Level Misconception Analysis for {rubric_title_display}\n\n" + "".join(analysis_results)
            return final_analysis
        else:
            return f"# Per-Level Misconception Analysis for {rubric_title_display}\n\nNo data available for analysis."

    except Exception as e:
        st.error(f"Critical error in setting up per-level analysis: {str(e)}")
        return f"### Per-Level Misconception Analysis\n\nError: Could not set up analysis. Details: {str(e)}"

# === MAIN STREAMLIT APPLICATION ===

def authenticate_user():
    """Simple authentication using session state"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        st.title("🎓 NCEA LLM Grading App")
        st.markdown("### Please enter your credentials to continue")
        
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Login"):
            if username == "admin" and password == "password":
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Invalid credentials. Please try again.")
        
        st.info("Default credentials: admin / password")
        return False
    
    return True

def load_rubric_files():
    """Load all available rubric files"""
    rubric_dir = Path("rubrics")
    if not rubric_dir.exists():
        st.error("Rubrics directory not found!")
        return {}
    
    rubrics = {}
    for rubric_file in rubric_dir.glob("*.json"):
        try:
            with open(rubric_file, 'r', encoding='utf-8') as f:
                rubric_data = json.load(f)
            rubrics[rubric_file.stem] = rubric_data
        except Exception as e:
            st.warning(f"Could not load rubric {rubric_file.name}: {e}")
    
    return rubrics

def log_grading_prompt(prompt: str, student_file: str):
    """Log the grading prompt to a file for evaluation purposes."""
    try:
        logging.debug(f"Attempting to log grading prompt for {student_file}")
        log_dir = Path("logs")
        if not log_dir.exists():
            log_dir.mkdir(exist_ok=True)
            logging.info(f"Created logs directory at {log_dir}")

        log_file = log_dir / "grading_prompts.log"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\n--- Grading Prompt for {student_file} ---\n")
            f.write(prompt)
            f.write("\n--- End of Prompt ---\n\n")
        logging.info(f"Successfully logged grading prompt for {student_file} to {log_file}")
    except Exception as e:
        logging.error(f"Failed to log grading prompt for {student_file}: {e}")

# Update process_student_portfolio to log the grading prompt
def process_student_portfolio(
    doc_path: Path, rubric_data: dict, openrouter_client, model_name: str, max_tokens: int
) -> Optional[Dict[str, Any]]:
    """Process a single student portfolio and return grading results."""
    # Validate file path based on extension
    if doc_path.suffix.lower() == '.pdf':
        if not validate_pdf_path(doc_path):
            return None
    elif doc_path.suffix.lower() != '.docx':
        st.error(f"Unsupported file type: {doc_path.suffix}")
        return None

    student_text = extract_text_from_file(doc_path)
    if not student_text:
        return None

    # Use smart truncation to preserve important content
    max_student_chars = 50000  # Increased from 12,000 to allow more content
    student_text = smart_truncate_text(student_text, max_student_chars, doc_path.name)

    prompt = build_grading_prompt(rubric_data, student_text)

    # If prompt is still too long, try more aggressive truncation
    if len(prompt) > 150000:  # Increased from 20,000 to 150,000 characters (~37K tokens)
        st.warning(f"Initial prompt too long for {doc_path.name}, applying more aggressive truncation...")
        
        # Try with even shorter text
        max_student_chars = 8000
        student_text = smart_truncate_text(student_text, max_student_chars, doc_path.name)
        
        prompt = build_grading_prompt(rubric_data, student_text)
        
        # Final check
        if len(prompt) > 150000:  # Increased from 20,000
            st.error(f"Prompt for {doc_path.name} is still too long after aggressive truncation. File may contain too much text.")
            return None

    # Log the grading prompt
    log_grading_prompt(prompt, doc_path.name)

    try:
        response = openrouter_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are an expert NCEA assessor. Analyze student work and provide detailed grading based on the rubric."},
                {"role": "user", "content": prompt}
            ],
            temperature=2.0,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )

        result_content = response.choices[0].message.content
        try:
            result = json.loads(result_content)
            result['student_file'] = doc_path.name
            result['file_path'] = str(doc_path)
            return result
        except json.JSONDecodeError as json_err:
            st.error(f"Invalid JSON response for {doc_path.name}: {str(json_err)}")
            logging.error(f"JSON decode error for {doc_path}: {json_err}")
            return None

    except Exception as api_err:
        logging.error(f"API error processing {doc_path}: {api_err}")
        st.error(f"Error processing {doc_path.name}: {api_err}")
        return None

def save_detailed_results_to_csv(detailed_results, filename="grading_detailed_results.csv"):
    """Save detailed grading results (including per_bullet if present) to CSV."""
    import pandas as pd
    rows = []
    for res in detailed_results:
        row = {
            'Student File': res.get('student_file', ''),
            'Grade': res.get('grade', ''),
            'Justification': res.get('justification', ''),
            'Failed Criteria': ", ".join(res.get('failed_criteria', [])),
            'Confidence Flags': ", ".join(res.get('confidence_flags', [])),
        }
        # Flatten per_bullet if present
        per_bullet = res.get('per_bullet', {})
        if isinstance(per_bullet, dict):
            for level, arr in per_bullet.items():
                row[f'per_bullet_{level}'] = str(arr)
        rows.append(row)
    df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode('utf-8')

# Add debugging logs to track `failed_criteria`
def process_portfolio(
    doc_path: Path, rubric_data: dict, client, model: str, max_tokens: int
) -> Optional[Dict[str, Any]]:
    """Process a single student portfolio and return grading results."""
    # Use cached text if available to avoid re-extraction
    cache_key = f"cached_text_{doc_path.name}"
    if cache_key in st.session_state:
        student_text = st.session_state[cache_key]
    else:
        student_text = extract_text_from_file(doc_path)
        if student_text:
            st.session_state[cache_key] = student_text
    
    if not student_text:
        return {}

    # Use smart truncation to preserve important content
    max_student_chars = 50000  # Increased from 12,000 to allow more content
    student_text = smart_truncate_text(student_text, max_student_chars, doc_path.name)

    prompt = build_grading_prompt(rubric_data, student_text)
    
    # If prompt is still too long, try more aggressive truncation
    if len(prompt) > 150000:  # Increased from 20,000 to 150,000 characters (~37K tokens)
        st.warning(f"Initial prompt too long for {doc_path.name}, applying more aggressive truncation...")
        
        # Try with even shorter text
        max_student_chars = 8000
        student_text = smart_truncate_text(student_text, max_student_chars, doc_path.name)
        
        prompt = build_grading_prompt(rubric_data, student_text)
        
        # Final check
        if len(prompt) > 150000:  # Increased from 20,000
            st.error(f"Prompt too long for {doc_path.name}")
            return {}

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert NCEA assessor."},
                {"role": "user", "content": prompt}
            ],
            temperature=2.0,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        
        # Log the full prompt length for debugging
        logging.info(f"Prompt length for {doc_path.name}: {len(prompt)} characters")
        
        try:
            result = json.loads(content)
        except json.JSONDecodeError as json_err:
            logging.error(f"JSON decode error for {doc_path.name}: {json_err}")
            logging.error(f"Raw content causing error: {content}")
            st.error(f"Invalid JSON response for {doc_path.name}: {str(json_err)}")
            return {}
        
        result['student_file'] = doc_path.name

        # Debug logging for grading results
        grade_assigned = result.get('grade', 'Unknown')
        logging.info(f"Grade assigned to {doc_path.name}: {grade_assigned}")
        
        # Debugging log for failed_criteria
        failed_criteria = result.get('failed_criteria', [])
        logging.info(f"Failed criteria for {doc_path.name}: {failed_criteria}")
        
        # Debug: Log a portion of the raw LLM response
        logging.info(f"Raw LLM response excerpt for {doc_path.name}: {content[:500]}...")

        return result
    except Exception as e:
        logging.error(f"API error for {doc_path.name}: {str(e)}")
        st.error(f"Grading failed for {doc_path.name}: {e}")
        return {}

def calculate_final_grade(grades: List[str]) -> str:
    """Calculate the final grade as a rough aggregate of three grades."""
    grade_counts = Counter(grades)
    most_common_grade, _ = grade_counts.most_common(1)[0]
    return most_common_grade

def calculate_variability(grades: List[str]) -> float:
    """Calculate variability as the number of unique grades divided by total grades."""
    unique_grades = len(set(grades))
    total_grades = len(grades)
    return unique_grades / total_grades

# Simplified main function with single-page layout
def main():
    """Main application function with simplified UI."""
    # Initialize session state and load custom CSS
    initialize_session_state()
    load_custom_css()
    
    # Debug: Check for unexpected app resets
    if "app_session_id" not in st.session_state:
        import uuid
        st.session_state.app_session_id = str(uuid.uuid4())
        logging.info(f"New app session started: {st.session_state.app_session_id}")
    
    # Check if we have previous results that should be restored
    if ("previous_results" in st.session_state and 
        st.session_state.previous_results and 
        "detailed_results" not in st.session_state):
        st.session_state["detailed_results"] = st.session_state.previous_results
        st.warning("Restored previous grading results after app reset.")
        logging.info("Restored previous results after app reset")
    
    with st.sidebar:
        st.title("NCEA LLM Grading App")
        if not authenticate_user():
            return
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.rerun()

        api_key = st.text_input("OpenRouter API Key", type="password", 
                               help="Enter your OpenRouter API key to access LLM models")
        max_tokens = st.slider("Max Tokens", 1000, 8000, 4000, 500,
                             help="Maximum tokens for LLM responses")
        
        # Show help sidebar
        show_help_sidebar()

    # Load rubrics and initialize client
    rubrics = load_rubric_files()
    if not rubrics:
        return
    
    # Show workflow progress
    show_workflow_progress()
    
    # === SINGLE PAGE LAYOUT === 
    st.markdown("---")
    
    # 1. Setup Section
    st.header("Assessment Configuration")
    col1, col2 = st.columns([2, 1])
    with col1:
        selected = st.selectbox(
            "Select Assessment Standard",
            list(rubrics.keys()),
            help="Choose the NCEA assessment standard for grading"
        )
    with col2:
        if st.button("Refresh Rubrics"):
            rubrics = load_rubric_files()
    
    rubric_data = rubrics[selected]

    if not api_key:
        st.warning("Please enter your OpenRouter API key in the sidebar to continue.")
        return

    try:
        client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    except Exception as e:
        st.error(f"Failed to initialize OpenRouter client: {e}")
        return

    # 2. File Upload Section
    st.header("Upload Student Portfolios")
    files = st.file_uploader(
        "Choose PDF/DOCX files or ZIP archives",
        type=['pdf', 'docx', 'zip'],
        accept_multiple_files=True,
        help="Upload individual PDF/DOCX portfolios or ZIP files containing multiple portfolios"
    )
    
    # Process files into supported documents
    documents = []
    if files:
        # Update file count in session state for workflow tracking
        st.session_state.uploaded_files_count = len(files)
        
        # Simple file validation
        total_size = sum(len(f.read()) for f in files)
        for f in files:
            f.seek(0)  # Reset file pointer
        
        if total_size > 100 * 1024 * 1024:  # 100MB
            st.warning(f"Large batch size ({total_size/(1024*1024):.1f}MB) - processing may take time")
        
        st.success(f"SUCCESS: {len(files)} file(s) uploaded")
        
        # Process files
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        
        for uploaded_file in files:
            file_ext = uploaded_file.name.lower()
            if file_ext.endswith('.pdf') or file_ext.endswith('.docx'):
                # Handle individual PDF or DOCX files
                doc_path = temp_dir / uploaded_file.name
                with open(doc_path, 'wb') as f:
                    f.write(uploaded_file.read())
                documents.append(doc_path)
            elif file_ext.endswith('.zip'):
                # Handle ZIP files
                zip_path = temp_dir / uploaded_file.name
                with open(zip_path, 'wb') as f:
                    f.write(uploaded_file.read())
                
                # Extract ZIP and find PDFs and DOCX files
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    extract_dir = temp_dir / f"extracted_{uploaded_file.name[:-4]}"
                    zip_ref.extractall(extract_dir)
                    
                    # Find all supported document types
                    for doc_file in extract_dir.rglob("*.pdf"):
                        documents.append(doc_file)
                    for doc_file in extract_dir.rglob("*.docx"):
                        documents.append(doc_file)
    else:
        # Reset file count when no files
        st.session_state.uploaded_files_count = 0

    # 3. Grading Section
    if documents:
        st.header("Start Grading Process")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.info(f"**Ready to grade**: {len(documents)} portfolios using {rubric_data.get('as_title', 'Unknown')}")
        with col2:
            grade_button = st.button("Start Grading", type="primary", use_container_width=True)
        
        # Handle grading process immediately in this section with progress shown here
        if 'grade_button' in locals() and grade_button and documents:
            # Reset previous results
            reset_grading_state()
            st.session_state.grading_started = True
            
            results = []
            
            # Show grading progress in this section
            st.markdown("---")
            st.subheader("Grading Progress")
            
            # Overall progress display
            overall_progress = st.progress(0)
            overall_status = st.empty()
            
            # Current student progress container
            current_student_container = st.container()
            
            total_files = len(documents)
            
            for file_idx, doc_file in enumerate(documents):
                try:
                    # Update overall progress
                    overall_progress.progress((file_idx) / total_files)
                    overall_status.text(f"Processing file {file_idx + 1} of {total_files}: {doc_file.name}")
                    
                    # Show only current student progress
                    with current_student_container:
                        st.markdown(f"**Current Student: {doc_file.name}**")
                        
                        # Progress for this student's grading attempts
                        student_progress = st.progress(0)
                        student_status = st.empty()
                        
                        grades = []
                        failed_criteria_list = []
                        
                        # Grade three times with detailed progress
                        for attempt in range(3):
                            try:
                                student_progress.progress((attempt) / 3)
                                student_status.text(f"Grading attempt {attempt + 1}/3...")
                                
                                result = process_portfolio(doc_file, rubric_data, client, "google/gemini-2.5-flash-preview-05-20", max_tokens)
                                if result:
                                    grades.append(result.get('grade', 'N/A'))
                                    failed_criteria_list.extend(result.get('failed_criteria', []))
                                    student_status.text(f"Attempt {attempt + 1}/3 complete - Grade: {result.get('grade', 'N/A')}")
                                else:
                                    grades.append('ERROR')
                                    student_status.text(f"Attempt {attempt + 1}/3 failed")
                            except Exception as attempt_error:
                                logging.error(f"Error in grading attempt {attempt + 1} for {doc_file.name}: {attempt_error}")
                                grades.append('ERROR')
                                student_status.text(f"Attempt {attempt + 1}/3 failed due to error")
                            
                            # Brief pause to show progress
                            time.sleep(0.1)
                        
                        # Complete student progress
                        student_progress.progress(1.0)
                        student_status.text("Processing detailed analysis...")
                        
                        if grades and any(g != 'ERROR' for g in grades):
                            # Get student text once (reuse from process_portfolio calls)
                            # Each process_portfolio call already extracted the text, so we can reuse it
                            # by extracting it once here for the detailed analysis
                            if f"cached_text_{doc_file.name}" not in st.session_state:
                                student_text = extract_text_from_file(doc_file)
                                st.session_state[f"cached_text_{doc_file.name}"] = student_text
                            else:
                                student_text = st.session_state[f"cached_text_{doc_file.name}"]
                            
                            if student_text:
                                failed_criteria_indepth = explain_failed_criteria_with_llm(
                                    openrouter_client=client,
                                    api_key=api_key,
                                    failed_criteria=failed_criteria_list,
                                    student_text=student_text,
                                    model_name="google/gemini-2.5-flash-preview-05-20",
                                    max_tokens=max_tokens
                                )
                            else:
                                failed_criteria_indepth = []

                            final_grade = calculate_final_grade([g for g in grades if g != 'ERROR'])
                            variability = calculate_variability([g for g in grades if g != 'ERROR'])
                            
                            results.append({
                                'Student File': doc_file.name,
                                'Grade 1': grades[0] if len(grades) > 0 else 'N/A',
                                'Grade 2': grades[1] if len(grades) > 1 else 'N/A',
                                'Grade 3': grades[2] if len(grades) > 2 else 'N/A',
                                'Final Grade': final_grade,
                                'Variability': variability,
                                'failed_criteria_indepth': json.dumps(failed_criteria_indepth, indent=2)
                            })
                            
                            # Save incremental results to prevent data loss
                            st.session_state["detailed_results"] = results
                            st.session_state["rubric"] = rubric_data
                            
                            student_status.text(f"COMPLETED - Final Grade: {final_grade}")
                        else:
                            student_status.text("FAILED - All grading attempts failed")
                
                except Exception as student_error:
                    # Handle individual student processing errors
                    logging.error(f"Error processing student {doc_file.name}: {student_error}")
                    st.error(f"Error processing {doc_file.name}: {str(student_error)}")
                    
                    # Add error result to maintain progress
                    results.append({
                        'Student File': doc_file.name,
                        'Grade 1': 'ERROR',
                        'Grade 2': 'ERROR', 
                        'Grade 3': 'ERROR',
                        'Final Grade': 'ERROR',
                        'Variability': 0.0,
                        'failed_criteria_indepth': json.dumps([f"Processing error: {str(student_error)}"], indent=2)
                    })
                
                # Clear the current student container to avoid crowding
                current_student_container.empty()
            
            # Complete overall progress
            overall_progress.progress(1.0)
            overall_status.text(f"Completed processing all {total_files} files")
            
            # Clear progress indicators
            time.sleep(1)  # Brief pause to show completion
            overall_progress.empty()
            overall_status.empty()
            current_student_container.empty()
            
            if results:
                st.success("Grading completed successfully!")
                st.balloons()
                
                # Store results in session state
                st.session_state["detailed_results"] = results
                st.session_state["rubric"] = rubric_data
                
                # Results will be displayed automatically in the Results Section below
                # No need for st.rerun() which would reset the UI
            else:
                st.error("No successful grading results. Please check your files and try again.")

    # 4. Results Section (Always visible, updates in real-time)
    st.header("Grading Results")
    
    # Results container that updates in real-time
    results_container = st.container()
    
    # Check if we have existing results
    if "detailed_results" in st.session_state and st.session_state["detailed_results"]:
        with results_container:
            display_results(st.session_state["detailed_results"])
    else:
        with results_container:
            st.info("No grading results yet. Upload files and start grading above.")

    # 5. Insights Section (Always visible)
    st.header("Learning Insights")
    
    if "detailed_results" in st.session_state and st.session_state["detailed_results"]:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info("Generate detailed learning insights and misconception analysis from grading results")
        with col2:
            insights_button = st.button("Generate Insights", type="secondary", use_container_width=True)
        
        if insights_button:
            generate_insights(api_key, max_tokens)
    else:
        st.info("Complete grading to generate learning insights.")

def display_results(results):
    """Display grading results with enhanced formatting."""
    df = pd.DataFrame(results)
    
    # Enhanced results display with metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Students", len(df))
    with col2:
        if 'Final Grade' in df.columns:
            avg_achievement = len(df[df['Final Grade'].isin(['A3', 'A4', 'M5', 'M6', 'E7', 'E8'])]) / len(df)
            st.metric("Achievement Rate", f"{avg_achievement:.1%}")
    with col3:
        if 'Variability' in df.columns:
            avg_variability = df['Variability'].mean()
            st.metric("Avg Variability", f"{avg_variability:.3f}")
    with col4:
        if 'Final Grade' in df.columns:
            mode_grade = df['Final Grade'].mode().iloc[0]
            st.metric("Most Common Grade", mode_grade)
    
    # Results table with download options
    st.subheader("Detailed Results")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        csv_data = df.to_csv(index=False).encode()
        st.download_button("Download CSV", csv_data, "grades.csv", "text/csv")
    with col2:
        report_text = export_detailed_report(df, st.session_state.get("rubric", {}))
        st.download_button("Download Report", report_text.encode(), "grading_report.md", "text/markdown")
    
    # Display the results table
    st.dataframe(df, use_container_width=True)

def generate_insights(api_key, max_tokens):
    """Generate learning insights and misconception analysis."""
    with st.spinner("Analyzing student work and generating insights..."):
        results = st.session_state.get("detailed_results")
        rubric = st.session_state.get("rubric")
        
        if not results or not rubric:
            st.error("Missing results or rubric data")
            return
        
        if not api_key:
            st.error("API key is required for insights generation")
            return
        
        try:
            client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
            df = pd.DataFrame(results)
            model_name = "google/gemini-2.5-flash-preview"
            max_tokens = 4000
            
            # Generate insights using the existing function
            insights = run_overall_misconception_analysis(
                openrouter_client=client,
                insights_df_source=df,
                llm_rubric_json=rubric,
                model_name=model_name,
                max_tokens=max_tokens
            )
            
            # Display insights
            st.success("Insights generated successfully!")
            st.markdown("---")
            st.markdown(insights, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error generating insights: {e}")

# Clean up unused variables (keep existing cleanup code)
fc_list = None
if isinstance(fc_list, str):
    try:
        fc_list = json.loads(fc_list)
    except json.JSONDecodeError:
        logging.error("Failed to deserialize failed_criteria_indepth JSON string.")
        pass

# Add custom CSS styling

# Custom CSS for better styling
def load_custom_css():
    """Load custom CSS for enhanced UI styling."""
    st.markdown("""
    <style>
    /* Main container styling */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Progress indicators */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Success messages */
    .success-banner {
        background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    
    /* Status cards */
    .status-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    
    /* File upload area enhancement */
    .uploadedFile {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: #f8f9fa;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 8px 8px 0 0;
        padding: 8px 16px;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #667eea;
        color: white;
    }
    
    /* Metric cards */
    [data-testid="metric-container"] {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Enhanced error handling and user guidance functions
def show_help_sidebar():
    """Display helpful information in sidebar."""
    with st.sidebar:
        st.markdown("---")
        st.subheader("Quick Help")
        
        with st.expander("Getting Started"):
            st.markdown("""
            1. **Enter API Key**: Add your OpenRouter API key
            2. **Select Rubric**: Choose the NCEA standard
            3. **Upload Files**: Add PDF/DOCX portfolios or ZIP files
            4. **Start Grading**: Begin automated assessment
            5. **View Results**: Check grades and statistics
            6. **Generate Insights**: Get learning analysis
            """)
        
        with st.expander("Troubleshooting"):
            st.markdown("""
            **Common Issues:**
            - **API Error**: Check your OpenRouter API key
            - **File Error**: Ensure files are readable PDFs or DOCX
            - **Slow Grading**: Large files take more time
            - **Memory Issues**: Try smaller batches
            
            **Tips:**
            - Use clear PDF files or properly formatted DOCX
            - Batch size: 5-10 files recommended
            - Check file sizes (< 10MB per file)
            - ZIP files can contain mixed PDF/DOCX files
            """)
        
        with st.expander("Understanding Results"):
            st.markdown("""
            **Supported File Types:**
            - PDF documents (.pdf)
            - Word documents (.docx)
            - ZIP archives containing PDFs and/or DOCX files
            
            **Grade Columns:**
            - **Grade 1-3**: Individual grading attempts
            - **Final Grade**: Consensus grade
            - **Variability**: Consistency measure (0-1)
            
            **Variability Guide:**
            - 0.0: Perfect agreement
            - 0.33: Some variation
            - 0.67+: High uncertainty
            """)

def validate_uploaded_files(files):
    """Validate uploaded files and provide feedback."""
    issues = []
    warnings = []
    
    total_size = 0
    pdf_count = 0
    
    for file in files:
        # Check file size
        file_size = len(file.read())
        file.seek(0)  # Reset file pointer
        total_size += file_size
        
        if file_size > 20 * 1024 * 1024:  # 20MB limit
            issues.append(f"⚠️ {file.name} is very large ({file_size/(1024*1024):.1f}MB)")
        
        # Check file type
        if file.name.lower().endswith('.pdf'):
            pdf_count += 1
        elif file.name.lower().endswith('.zip'):
            # Could contain multiple PDFs
            pass
        else:
            issues.append(f"❌ {file.name} is not a supported format")
    
    # Overall checks
    if total_size > 100 * 1024 * 1024:  # 100MB total
        warnings.append(f"📊 Large batch size ({total_size/(1024*1024):.1f}MB total) - consider smaller batches")
    
    if len(files) > 20:
        warnings.append(f"📈 Many files ({len(files)}) - grading may take significant time")
    
    return issues, warnings

def show_status_updates(current_file, progress, total_files, current_step=""):
    """Show real-time status updates during processing."""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.write(f"📄 Processing: **{current_file}**")
        if current_step:
            st.caption(f"Step: {current_step}")
    
    with col2:
        st.write(f"{progress}/{total_files}")
    
    # Progress bar with percentage
    progress_pct = progress / total_files if total_files > 0 else 0
    st.progress(progress_pct, text=f"Overall Progress: {progress_pct*100:.0f}%")

# Enhanced session state management
def initialize_session_state():
    """Initialize session state variables for better UX."""
    if "grading_started" not in st.session_state:
        st.session_state.grading_started = False
    
    if "current_step" not in st.session_state:
        st.session_state.current_step = "setup"
    
    if "uploaded_files_count" not in st.session_state:
        st.session_state.uploaded_files_count = 0
    
    if "grading_progress" not in st.session_state:
        st.session_state.grading_progress = {"completed": 0, "total": 0}
    
    if "last_rubric" not in st.session_state:
        st.session_state.last_rubric = None
    
    if "insights_generated" not in st.session_state:
        st.session_state.insights_generated = False

def update_progress_state(completed, total, current_file=""):
    """Update grading progress in session state."""
    st.session_state.grading_progress = {
        "completed": completed,
        "total": total,
        "current_file": current_file
    }

def reset_grading_state():
    """Reset grading-related session state when starting new batch."""
    logging.info("Resetting grading state - preserving previous results")
    st.session_state.grading_started = False
    st.session_state.grading_progress = {"completed": 0, "total": 0}
    st.session_state.insights_generated = False
    # Don't automatically delete detailed_results - only clear them if we successfully complete new grading
    # This prevents losing results if something goes wrong during grading
    if "previous_results" not in st.session_state:
        st.session_state.previous_results = st.session_state.get("detailed_results", [])

def get_workflow_status():
    """Get current workflow status for UI guidance."""
    status = {
        "setup_complete": False,
        "files_uploaded": False,
        "grading_complete": False,
        "insights_available": False
    }
    
    # Check if basic setup is complete
    status["setup_complete"] = (
        "authenticated" in st.session_state and 
        st.session_state.authenticated
    )
    
    # Check if files are uploaded
    status["files_uploaded"] = st.session_state.uploaded_files_count > 0
    
    # Check if grading is complete
    status["grading_complete"] = (
        "detailed_results" in st.session_state and 
        st.session_state["detailed_results"]
    )
    
    # Check if insights are available
    status["insights_available"] = status["grading_complete"]
    
    return status

def show_workflow_progress():
    """Display workflow progress indicator."""
    status = get_workflow_status()
    
    st.markdown("### Workflow Progress")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        icon = "DONE" if status["setup_complete"] else "PENDING"
        st.markdown(f"**{icon}: Setup**")
        st.caption("Authentication & Configuration")
    
    with col2:
        icon = "DONE" if status["files_uploaded"] else "PENDING"
        st.markdown(f"**{icon}: Upload**")
        st.caption("Portfolio Files")
    
    with col3:
        icon = "DONE" if status["grading_complete"] else "PENDING"
        st.markdown(f"**{icon}: Grading**")
        st.caption("Automated Assessment")
    
    with col4:
        icon = "DONE" if status["insights_available"] else "PENDING"
        st.markdown(f"**{icon}: Insights**")
        st.caption("Learning Analysis")
    
    # Progress bar - count only boolean True values
    completed_steps = sum(1 for value in status.values() if value is True)
    progress = completed_steps / 4
    st.progress(progress, text=f"Workflow: {completed_steps}/4 steps complete")

# Enhanced data visualization functions
def create_grade_distribution_chart(df):
    """Create an enhanced grade distribution visualization."""
    if 'Final Grade' not in df.columns:
        return None
    
    grade_counts = df['Final Grade'].value_counts()
    
    # Create a properly ordered chart
    ordered_grades = [grade for grade in GRADE_ORDER if grade in grade_counts.index]
    ordered_counts = [grade_counts[grade] for grade in ordered_grades]
    
    chart_df = pd.DataFrame({
        'Grade': ordered_grades,
        'Count': ordered_counts
    })
    
    return chart_df

def create_variability_analysis(df):
    """Analyze grading variability and provide insights."""
    if 'Variability' not in df.columns:
        return None
    
    variability_stats = {
        'mean': df['Variability'].mean(),
        'median': df['Variability'].median(),
        'std': df['Variability'].std(),
        'high_variability': len(df[df['Variability'] > 0.5]),
        'perfect_agreement': len(df[df['Variability'] == 0.0])
    }
    
    return variability_stats

def export_detailed_report(df, rubric_data):
    """Generate a detailed exportable report."""
    report_lines = []
    
    # Header
    report_lines.extend([
        "# NCEA Grading Report",
        f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Assessment: {rubric_data.get('as_title', 'Unknown')}",
        f"Standard: {rubric_data.get('as_code', 'Unknown')}",
        "",
        "## Summary Statistics"
    ])
    
    # Summary stats
    total_students = len(df)
    if 'Final Grade' in df.columns:
        grade_counts = df['Final Grade'].value_counts()
        achieved_count = len(df[~df['Final Grade'].isin(['N1', 'N2'])])
        achievement_rate = (achieved_count / total_students * 100) if total_students > 0 else 0
        
        report_lines.extend([
            f"- Total Students: {total_students}",
            f"- Achievement Rate: {achievement_rate:.1f}%",
            f"- Most Common Grade: {grade_counts.index[0] if len(grade_counts) > 0 else 'N/A'}",
            ""
        ])
    
    # Grade distribution
    if 'Final Grade' in df.columns:
        report_lines.extend(["## Grade Distribution", ""])
        for grade in GRADE_ORDER:
            if grade in grade_counts:
                percentage = (grade_counts[grade] / total_students * 100)
                report_lines.append(f"- {grade}: {grade_counts[grade]} students ({percentage:.1f}%)")
    
    # Variability analysis
    if 'Variability' in df.columns:
        var_stats = create_variability_analysis(df)
        if var_stats:
            report_lines.extend([
                "",
                "## Grading Consistency Analysis",
                f"- Average Variability: {var_stats['mean']:.3f}",
                f"- Median Variability: {var_stats['median']:.3f}",
                f"- High Variability (>0.5): {var_stats['high_variability']} students",
                f"- Perfect Agreement: {var_stats['perfect_agreement']} students",
            ])
    
    return "\n".join(report_lines)

if __name__ == "__main__":
    main()
