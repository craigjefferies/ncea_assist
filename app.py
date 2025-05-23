import streamlit as st
import os
import fitz  # PyMuPDF
from bs4 import BeautifulSoup
import logging
import json
import re # For parsing JSON from markdown code blocks
from openrouter import OpenRouter, OpenRouterError
from pathlib import Path
import tempfile
import zipfile
import asyncio
import pandas as pd
from collections import Counter
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
GRADE_ORDER = ["Not Achieved", "Achieved", "Merit", "Excellence"]
MAX_STUDENT_CHARS_FOR_LLM = 30000
MAX_API_PROMPT_CHARS = 40000
DEFAULT_RUBRICS_DIR = "rubrics"
DEFAULT_CACHE_DIR = "cv_cache" # For local directory selection (was cache)
TEMP_UPLOAD_DIR_NAME = "temp_cv_uploads" # Base name for temp folder within system temp

# --- Helper Functions (Existing or Modified/New) ---

def extract_text_from_pdf(file_path):
    """
    Extracts text from a PDF file, attempting both raw text extraction and
    HTML conversion for potentially richer content.
    Includes improved error handling, image detection, and text quality heuristics.
    Returns extracted text, a list of warnings, and a boolean indicating if images
    were found on pages with minimal text.
    """
    raw_text = ""
    html_text = ""
    cleaned_html_text = ""
    warnings = []
    has_images_on_low_text_pages = False

    try:
        doc = fitz.open(file_path)
    except fitz.fitz.FitzError as e:
        error_message = f"Error opening PDF (fitz.fitz.FitzError): {e}. The file might be corrupted, password-protected, or not a valid PDF."
        logging.error(error_message)
        warnings.append(error_message)
        return "", warnings, False
    except Exception as e:
        error_message = f"Unexpected error opening PDF with fitz: {e}."
        logging.error(error_message)
        warnings.append(error_message)
        return "", warnings, False

    try:
        if not doc.is_pdf:
            warn_msg = "Warning: The file does not appear to be a standard PDF. Attempting to process anyway."
            warnings.append(warn_msg)
            logging.warning(warn_msg)
        if doc.needs_pass:
            error_msg = "Error: PDF is password-protected. Text extraction cannot proceed."
            warnings.append(error_msg)
            logging.error(error_msg)
            doc.close()
            return "", warnings, False
        total_pages = doc.page_count
        if total_pages == 0:
            warn_msg = "Warning: PDF document has no pages."
            warnings.append(warn_msg)
            logging.warning(warn_msg)
            doc.close()
            return "", warnings, False

        for page_num in range(total_pages):
            page = doc.load_page(page_num)
            page_raw_text = page.get_text("text").strip()
            raw_text += page_raw_text + "\n"
            try:
                page_html_content = page.get_text("html")
                html_text += page_html_content + "\n"
            except Exception as e:
                logging.warning(f"Could not extract HTML content from page {page_num + 1}: {e}")
                warnings.append(f"Note: Could not extract structured text (HTML) from page {page_num + 1}.")
            if len(page_raw_text) < 100 and page.get_images(full=True):
                has_images_on_low_text_pages = True
                logging.info(f"Page {page_num + 1} has minimal text and contains images.")
        doc.close()

        if html_text:
            try:
                soup = BeautifulSoup(html_text, 'html.parser')
                for script_or_style in soup(["script", "style"]):
                    script_or_style.decompose()
                cleaned_html_text = soup.get_text(separator="\n", strip=True)
            except Exception as e:
                logging.error(f"Error parsing HTML content with BeautifulSoup: {e}")
                warnings.append("Warning: Could not fully parse structured text (HTML).")
                cleaned_html_text = ""
        
        final_text = raw_text.strip()
        raw_text_effectively_empty = len(raw_text.strip()) < 20
        html_text_effectively_empty = len(cleaned_html_text.strip()) < 20

        if raw_text_effectively_empty and html_text_effectively_empty:
            warn_msg = "Warning: No significant text content could be extracted. "
            warn_msg += "Images were detected, suggesting scanned document." if has_images_on_low_text_pages else "May be image-only or empty."
            warnings.append(warn_msg)
        elif not html_text_effectively_empty and \
             (len(cleaned_html_text) > len(raw_text) * 1.2 or \
             (len(raw_text.strip()) < 200 and len(cleaned_html_text) > len(raw_text.strip()))):
            final_text = cleaned_html_text
            logging.info("Chose HTML extracted text.")
        else:
            logging.info("Chose raw extracted text.")
            if not html_text_effectively_empty and len(cleaned_html_text) < len(raw_text) * 0.5:
                 warnings.append("Warning: HTML text shorter than raw, using raw.")

        if not final_text.strip() and not any("Error:" in w for w in warnings) and not any("Warning: No significant text" in w for w in warnings):
            warn_msg = "Warning: Extracted text is empty or whitespace."
            if has_images_on_low_text_pages: warn_msg += " Images detected, consider OCR."
            warnings.append(warn_msg)
        if has_images_on_low_text_pages and not any("Images were detected" in w for w in warnings) and not any("minimal text content also contain images" in w for w in warnings):
             warnings.append("Note: Images on low-text pages detected. OCR might be needed if text is missing.")
        return final_text.strip(), list(set(warnings)), has_images_on_low_text_pages
    except Exception as e:
        logging.error(f"Unexpected error in PDF extraction: {e}", exc_info=True)
        warnings.append(f"Unexpected PDF processing error: {e}")
        return raw_text.strip() if raw_text.strip() else "", warnings, has_images_on_low_text_pages

def build_grading_prompt(rubric_data, cv_text):
    if not isinstance(rubric_data, dict):
        logging.error("Invalid rubric_data: not a dictionary.")
        return "Error: Rubric data is not in the expected format."
    as_code = rubric_data.get('as_code', 'Unknown')
    as_title = rubric_data.get('as_title', 'Unknown Title')
    criteria_section_parts = []
    grading_criteria_list = rubric_data.get('gradingCriteria', [])
    if not grading_criteria_list:
        criteria_section = "No specific grading criteria provided in the rubric."
    else:
        for level_info in grading_criteria_list:
            if not isinstance(level_info, dict):
                logging.warning(f"Skipping malformed level_info: {level_info}")
                continue
            level_name = level_info.get('levelName', 'Unnamed Level')
            sublevels_list = level_info.get('sublevels', [])
            sublevels_str = ""
            if sublevels_list and isinstance(sublevels_list, list):
                valid_sublevels = [str(sl) for sl in sublevels_list if sl]
                if valid_sublevels: sublevels_str = f" ({', '.join(valid_sublevels)})"
            main_requirement = level_info.get('mainRequirement', 'N/A')
            criteria_involves_list = level_info.get('criteriaInvolves', [])
            involved_criteria_str_parts = []
            if criteria_involves_list and isinstance(criteria_involves_list, list):
                for criterion in criteria_involves_list:
                    if criterion and isinstance(criterion, str):
                        involved_criteria_str_parts.append(f"    - {criterion.strip()}")
            criteria_detail = f"### {level_name}{sublevels_str}:\n- Main Requirement: {main_requirement}\n"
            criteria_detail += "- Criteria Involves:\n" + "\n".join(involved_criteria_str_parts) if involved_criteria_str_parts else "- Criteria Involves: N/A"
            criteria_section_parts.append(criteria_detail)
        criteria_section = "\n\n".join(criteria_section_parts)
    prompt = f"""You are an AI assistant specialized in evaluating CVs based on specific rubrics.
Your task is to analyze the provided CV text against the grading criteria outlined below.
Rubric Code: {as_code}
Rubric Title: {as_title}
Grading Criteria:
{criteria_section}
CV Text to Analyze:
--- BEGIN CV ---
{cv_text}
--- END CV ---
Instructions:
1. For each criterion in the "Grading Criteria" section, assess how well the CV meets the requirements.
2. Provide a detailed evaluation for each criterion, explaining your reasoning.
3. If applicable, suggest a score or rating for each criterion based on its definition.
4. Conclude with an overall assessment of the CV against the rubric.
Provide your evaluation in a structured format. Ensure your entire response is a single JSON object with keys: "grade", "justification", "failed_criteria" (a list of strings, where each string is the 'Main Requirement' of a criterion that was not met or only partially met)."""
    return prompt.strip()

async def process_student_portfolio(
    student_text: str, pdf_path: Path, llm_rubric_json: dict, 
    api_key: str, model_name: str, temperature: float, openrouter_client: OpenRouter
) -> dict:
    result = {
        'student_file': pdf_path.name, 'grade': 'Processing Error', 
        'justification': 'An unexpected issue occurred.', 'failed_criteria': [], 
        'confidence_flags': [], 'error_stage': 'unknown', 
        'model_used': model_name, 'llm_temperature': temperature,
        'raw_llm_response': '', 'per_bullet_runs_aggregation': {} # Added for CSV
    }
    if student_text is None or not student_text.strip():
        logging.warning(f"Text extraction failed for {result['student_file']}.")
        result.update({'grade': 'Extraction Failed', 'justification': 'Could not extract text from PDF.', 
                       'error_stage': 'text_extraction', 'confidence_flags': ['Text extraction error']})
        return result

    student_text_for_llm = student_text
    if len(student_text) > MAX_STUDENT_CHARS_FOR_LLM:
        original_len = len(student_text)
        student_text_for_llm = student_text[:MAX_STUDENT_CHARS_FOR_LLM] + "\n\n... [Student work truncated...]"
        logging.info(f"Student text for {result['student_file']} truncated: {original_len} to {len(student_text_for_llm)}.")
        result['confidence_flags'].append(f"Student text truncated from {original_len} to {MAX_STUDENT_CHARS_FOR_LLM} chars.")

    prompt_for_llm = build_grading_prompt(llm_rubric_json, student_text_for_llm)
    if prompt_for_llm.startswith("Error: Rubric data"):
        logging.error(f"Prompt generation failed for {result['student_file']}: {prompt_for_llm}")
        result.update({'grade': 'Prompt Error', 'justification': prompt_for_llm, 'error_stage': 'prompt_generation', 
                       'confidence_flags': result['confidence_flags'] + ['Rubric data error']})
        return result
    if len(prompt_for_llm) > MAX_API_PROMPT_CHARS:
        logging.error(f"Prompt for {result['student_file']} exceeds MAX_API_PROMPT_CHARS.")
        result.update({'grade': 'Prompt Error', 'justification': f"Prompt too long ({len(prompt_for_llm)} chars).", 
                       'error_stage': 'prompt_generation', 'confidence_flags': result['confidence_flags'] + ['Prompt too long']})
        return result

    try:
        logging.info(f"Sending LLM request for {result['student_file']}. Model: {model_name}, Temp: {temperature}.")
        response = await openrouter_client.chat.completions.create(
            model=model_name, messages=[{"role": "user", "content": prompt_for_llm}],
            temperature=temperature, max_tokens=2000
        )
        llm_response_content = response.choices[0].message.content.strip()
        result['raw_llm_response'] = llm_response_content # Store raw response
        logging.info(f"LLM response for {result['student_file']} (len {len(llm_response_content)}).")
    except OpenRouterError as e:
        logging.error(f"OpenRouter API error for {result['student_file']}: {e}", exc_info=True)
        err_str, user_msg = str(e).lower(), f"Generic API error: {e}"
        if "401" in err_str or "authentication" in err_str:
            if 'st' in globals() and hasattr(st, 'session_state'): st.session_state['api_key_valid'] = False
            user_msg = "Authentication error: Check API key."
        elif "429" in err_str or "rate limit" in err_str: user_msg = "API rate limit exceeded."
        elif "context length" in err_str or "token limit" in err_str: user_msg = "Input too long for model."
        elif any(c in err_str for c in ["500","502","503"]) or "server error" in err_str: user_msg = "OpenRouter server error."
        result.update({'grade': 'API Error', 'justification': user_msg, 'error_stage': 'llm_api_call', 
                       'confidence_flags': result['confidence_flags'] + ['API call failed']})
        return result
    except Exception as e:
        logging.error(f"Unexpected API error for {result['student_file']}: {e}", exc_info=True)
        result.update({'grade': 'API Error', 'justification': f"Unexpected API error: {e}", 'error_stage': 'llm_api_call', 
                       'confidence_flags': result['confidence_flags'] + ['Unexpected API error']})
        return result

    match = re.search(r"```json\s*([\s\S]*?)\s*```", llm_response_content, re.DOTALL)
    cleaned_json_str = match.group(1).strip() if match else llm_response_content.strip()
    try:
        parsed_llm_response = json.loads(cleaned_json_str)
    except json.JSONDecodeError as e:
        logging.error(f"JSONDecodeError for {result['student_file']}. Snippet: {cleaned_json_str[:500]}", exc_info=True)
        result.update({'grade': 'LLM Format Error', 'justification': f"Invalid JSON. Error: {e}. Snippet: '{cleaned_json_str[:200]}...'", 
                       'error_stage': 'llm_response_parsing', 'confidence_flags': result['confidence_flags'] + ['Invalid JSON from LLM']})
        return result

    if isinstance(parsed_llm_response, dict):
        result['grade'] = parsed_llm_response.get('grade', 'Ungraded')
        result['justification'] = parsed_llm_response.get('justification', 'No justification from LLM.')
        result['failed_criteria'] = parsed_llm_response.get('failed_criteria', [])
        result['error_stage'] = None
        if result['grade'] == 'Ungraded': result['confidence_flags'].append("LLM missing 'grade'.")
        if result['justification'] == 'No justification from LLM.': result['confidence_flags'].append("LLM missing 'justification'.")
        if not isinstance(result['failed_criteria'], list):
            logging.warning(f"LLM 'failed_criteria' not a list for {result['student_file']}.")
            result['failed_criteria'] = []
            result['confidence_flags'].append("LLM 'failed_criteria' not a list.")
        logging.info(f"Successfully processed {result['student_file']}. Grade: {result['grade']}")
    else:
        logging.error(f"LLM response for {result['student_file']} not a dict. Type: {type(parsed_llm_response)}.")
        result.update({'grade': 'LLM Format Error', 'justification': f"LLM response not a dict. Type: {type(parsed_llm_response)}", 
                       'error_stage': 'llm_response_parsing', 'confidence_flags': result['confidence_flags'] + ['LLM response not dict']})
    return result

async def explain_failed_criteria_with_llm(
    failed_criteria: list, cv_text: str, rubric_text: str, 
    api_key: str, model_name: str, openrouter_client: OpenRouter
) -> list:
    formatted_explanations = [{"criterion": fc, "explanation": "API key not configured or explanation failed."} for fc in failed_criteria]
    if not failed_criteria or not api_key: return formatted_explanations
    
    failed_criteria_str = "\n".join([f"- \"{crit}\"" for crit in failed_criteria])
    prompt_text = f"""CV Text:\n{cv_text}\n\nRubric Text:\n{rubric_text}\n\nFailed Criteria:\n{failed_criteria_str}\n
Your task: return a JSON list of objects, each with "criterion" (exact text from above) and "explanation" (why it failed).
Example: [{{"criterion": "Failed criterion text 1", "explanation": "Explanation..."}}]
Ensure valid JSON list. Match number of objects to criteria."""
    try:
        logging.info(f"Requesting explanation for {len(failed_criteria)} criteria using {model_name}.")
        response = await openrouter_client.chat.completions.create(
            model=model_name, messages=[{"role": "user", "content": prompt_text}],
            temperature=0.5, max_tokens=150 * len(failed_criteria) + 300
        )
        result_json_str = response.choices[0].message.content.strip()
        logging.info(f"LLM explanations raw response: {result_json_str[:300]}...")
        parsed_json_data = json.loads(result_json_str)
        if isinstance(parsed_json_data, list):
            llm_explanations_list = parsed_json_data
        elif isinstance(parsed_json_data, dict) and "explanations" in parsed_json_data and isinstance(parsed_json_data["explanations"], list):
            llm_explanations_list = parsed_json_data["explanations"]
        else: # fallback or error
            logging.warning("LLM explanations response has unexpected structure.")
            return formatted_explanations # return default error explanations

        # Update formatted_explanations based on llm_explanations_list
        for i, fc_orig in enumerate(failed_criteria):
            if i < len(llm_explanations_list) and isinstance(llm_explanations_list[i], dict):
                # Prioritize matching by original criterion text if possible, else by order
                llm_item = llm_explanations_list[i]
                # Simple update by order for this version
                formatted_explanations[i]["explanation"] = llm_item.get("explanation", "No explanation from LLM.")
                if llm_item.get("criterion") != fc_orig:
                     logging.warning(f"LLM explanation criterion mismatch: Original '{fc_orig}', LLM '{llm_item.get('criterion')}'. Used explanation anyway.")
            else:
                formatted_explanations[i]["explanation"] = "Malformed or missing explanation from LLM."
        return formatted_explanations
    except (OpenRouterError, json.JSONDecodeError, Exception) as e:
        logging.error(f"Error explaining criteria: {e}", exc_info=True)
        # Keep default error explanations
        return formatted_explanations


async def run_overall_misconception_analysis(
    insights_df_source, llm_rubric_json: dict, grade_distribution: dict, 
    common_misconceptions: list, api_key: str, model_name: str, openrouter_client: OpenRouter
) -> str:
    all_explanations_texts = []
    # Simplified data check: assume main() prepares insights_df_source appropriately
    if not insights_df_source or 'failed_criteria_indepth' not in insights_df_source[0]: # Basic check
        return "### Overall Misconception Analysis\n\nError: Data missing for analysis."
    for row_data in insights_df_source:
        fc_list = row_data.get('failed_criteria_indepth', [])
        if isinstance(fc_list, str): 
            try: fc_list = json.loads(fc_list)
            except: fc_list = []
        if isinstance(fc_list, list):
            for item in fc_list:
                if isinstance(item, dict) and item.get('explanation'):
                    all_explanations_texts.append(item['explanation'].strip())
    if not all_explanations_texts: return "### Overall Misconception Analysis\n\nNo explanations for analysis."

    rubric_as_code = llm_rubric_json.get('as_code', 'UNKNOWN')
    rubric_as_title = llm_rubric_json.get('as_title', 'Unknown Title')
    rubric_title_display = f"AS{rubric_as_code} ('{rubric_as_title}')" if rubric_as_code != 'UNKNOWN' and rubric_as_title != 'Unknown Title' else "The Standard"
    analysis_title_md = f"### Overall Misconception Analysis for {rubric_title_display}\n\n"
    
    criteria_summary = "\n".join([f"- '{c['criterion']}' ({c['percentage']:.0f}%)" for c in common_misconceptions]) if common_misconceptions else "None."
    grade_dist_summary = "\n".join([f"- {g}: {c} CVs" for g,c in grade_distribution.items()])
    unique_explanations = "\n".join(list(set(all_explanations_texts)))
    
    prompt = f"""Expert analysis for {rubric_title_display}:
Commonly failed criteria:\n{criteria_summary}
Grade Distribution:\n{grade_dist_summary}
Explanations for failures:\n{unique_explanations}\n
Synthesize a report (3-5 paragraphs): key themes, reasons, recommendations."""
    try:
        response = await openrouter_client.chat.completions.create(
            model=model_name, messages=[{"role": "user", "content": prompt}],
            temperature=0.6, max_tokens=800
        )
        return analysis_title_md + response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Overall analysis LLM error: {e}", exc_info=True)
        return analysis_title_md + f"Error during analysis: {e}"

async def run_per_level_misconception_analysis(
    insights_df_source, llm_rubric_json: dict, common_misconceptions_by_level: dict, 
    api_key: str, model_name: str, openrouter_client: OpenRouter
) -> str:
    rubric_as_code = llm_rubric_json.get('as_code', 'UNKNOWN')
    rubric_as_title = llm_rubric_json.get('as_title', 'Unknown Title')
    rubric_title_display = f"AS{rubric_as_code} ('{rubric_as_title}')" if rubric_as_code != 'UNKNOWN' and rubric_as_title != 'Unknown Title' else "The Standard"
    main_title = f"# Per-Level Misconception Analysis for {rubric_title_display}\n\n"
    level_parts = []

    if not insights_df_source or 'failed_criteria_indepth' not in insights_df_source[0]: # Basic check
        return main_title + "Error: Data missing for per-level analysis."

    for level in GRADE_ORDER:
        level_md = f"## {level} Level\n\n"
        level_data = [r for r in insights_df_source if r.get('grade') == level]
        if not level_data:
            level_md += "No students at this level or data unavailable.\n\n"
            level_parts.append(level_md)
            continue
        
        explanations = []
        for row in level_data:
            fc_list = row.get('failed_criteria_indepth', [])
            if isinstance(fc_list, str): 
                try: fc_list = json.loads(fc_list)
                except: fc_list = []
            if isinstance(fc_list, list):
                for item in fc_list:
                    if isinstance(item, dict) and item.get('explanation'):
                        explanations.append(item['explanation'].strip())
        if not explanations:
            level_md += "No explanations found for this level.\n\n"
            level_parts.append(level_md)
            continue

        common_issues = common_misconceptions_by_level.get(level, [])
        criteria_summary = "\n".join([f"- '{c['criterion']}' ({c['percentage']:.0f}%)" for c in common_issues]) if common_issues else "None."
        unique_explanations = "\n".join(list(set(explanations)))
        prompt = f"""Expert analysis for '{level}' grade students of {rubric_title_display}:
Commonly failed criteria for this level:\n{criteria_summary}
Explanations for failures at this level:\n{unique_explanations}\n
Synthesize a report (2-4 paragraphs) for this level: themes, reasons, recommendations."""
        try:
            response = await openrouter_client.chat.completions.create(
                model=model_name, messages=[{"role": "user", "content": prompt}],
                temperature=0.55, max_tokens=600
            )
            level_md += response.choices[0].message.content.strip() + "\n\n"
        except Exception as e:
            logging.error(f"Per-level analysis LLM error for {level}: {e}", exc_info=True)
            level_md += f"Error during analysis for this level: {e}\n\n"
        level_parts.append(level_md)
    return main_title + "".join(level_parts)

# --- Refactored Helper Functions ---
def authenticate_user():
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False
    
    APP_USER = os.environ.get("APP_USERNAME", "admin")
    APP_PASS = os.environ.get("APP_PASSWORD", "password")

    if st.session_state['authenticated']:
        if st.sidebar.button("Logout", key="logout_button"):
            st.session_state['authenticated'] = False
            # Clear relevant session state keys
            keys_to_clear = ['api_key', 'student_files_to_process', 
                             'cached_detailed_results_for_insights', 'temp_upload_dir',
                             'rubric_data', 'selected_rubric_name', 'api_key_valid',
                             'grading_llm_model', 'grading_temperature', 
                             'analysis_llm_model', 'analysis_temperature', 'max_output_tokens',
                             'num_grading_runs']
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            if 'openrouter_client' in st.session_state: # specific client object
                del st.session_state['openrouter_client']
            st.sidebar.success("Logged out successfully.")
            st.rerun() # Use rerun to refresh the UI state after logout
        return True

    with st.sidebar.form("login_form"):
        st.write("Please login")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        submitted = st.form_submit_button("Login")

        if submitted:
            if username == APP_USER and password == APP_PASS:
                st.session_state['authenticated'] = True
                st.sidebar.success("Logged in successfully!")
                st.rerun() # Rerun to update UI post-login
            else:
                st.sidebar.error("Incorrect username or password.")
    return False

def load_rubric_files(rubrics_dir_path_str: str = DEFAULT_RUBRICS_DIR) -> dict:
    rubrics_dir = Path(rubrics_dir_path_str)
    if not rubrics_dir.exists():
        try:
            rubrics_dir.mkdir(parents=True, exist_ok=True)
            st.info(f"Rubrics directory created at '{rubrics_dir.resolve()}'. Please add your rubric JSON files there.")
        except Exception as e:
            st.error(f"Error creating rubrics directory '{rubrics_dir}': {e}")
            return {} # Return empty if dir creation fails
        return {}

    rubric_files = list(rubrics_dir.glob("*.json"))
    if not rubric_files:
        st.warning(f"No rubric files (*.json) found in '{rubrics_dir.resolve()}'.")
        return {}

    loaded_rubrics = {}
    for rubric_file in rubric_files:
        try:
            with open(rubric_file, 'r', encoding='utf-8') as f:
                rubric_data = json.load(f)
            
            # Validate essential keys
            essential_keys = ['as_code', 'as_title', 'gradingCriteria']
            if not all(key in rubric_data for key in essential_keys):
                st.warning(f"Skipping rubric '{rubric_file.name}': Missing one or more essential keys ({', '.join(essential_keys)}).")
                continue
            if not isinstance(rubric_data['gradingCriteria'], list):
                 st.warning(f"Skipping rubric '{rubric_file.name}': 'gradingCriteria' must be a list.")
                 continue

            rubric_name = rubric_data.get('as_title', rubric_file.stem) # Use as_title or filename stem
            loaded_rubrics[rubric_name] = rubric_data
            logging.info(f"Successfully loaded rubric: {rubric_name} from {rubric_file.name}")

        except json.JSONDecodeError as e:
            st.warning(f"Skipping rubric '{rubric_file.name}': Invalid JSON format. Error: {e}")
        except Exception as e:
            st.warning(f"Skipping rubric '{rubric_file.name}': Error loading file. Error: {e}")
            
    return loaded_rubrics

def save_detailed_results_to_csv(results_list: list, filename_prefix: str, selected_rubric_name: str) -> str:
    if not results_list:
        return ""
    
    # Flatten the results for CSV
    flat_results = []
    for result in results_list:
        flat_row = {
            'Student File': result.get('student_file', 'N/A'),
            'Final Assigned Grade': result.get('final_grade_assigned', 'N/A'), # Added
            'LLM Grade (First Run)': result.get('grade', 'N/A'), # Grade from the canonical (first successful/first) run
            'Justification (First Run)': result.get('justification', 'N/A'),
            'Model Used (First Run)': result.get('model_used', 'N/A'),
            'Temperature (First Run)': result.get('llm_temperature', 'N/A'),
            'Error Stage (First Run)': result.get('error_stage', 'None'),
            'Raw LLM Response (First Run)': result.get('raw_llm_response', ''), # if error, this might be empty
            'Confidence Flags': ', '.join(result.get('confidence_flags', [])),
        }
        
        # Flatten failed_criteria from the canonical run
        failed_criteria_list = result.get('failed_criteria', [])
        if isinstance(failed_criteria_list, list):
             flat_row['Failed Criteria (First Run)'] = '; '.join(failed_criteria_list) if failed_criteria_list else 'None'
        else: # Should not happen with new process_student_portfolio, but good for safety
             flat_row['Failed Criteria (First Run)'] = str(failed_criteria_list)


        # Flatten per_bullet_runs_aggregation
        aggregation = result.get('per_bullet_runs_aggregation', {})
        # Assuming aggregation structure like: {'Criterion Name': {'Run 1': 'A', 'Run 2': 'M'}, ...}
        # This needs to be more dynamic or assume max runs for columns
        # For simplicity, let's assume we just dump the dict as string for now, or pick specific runs.
        # A better approach would be to know max_runs and create columns like Criterion1_Run1_Grade, Criterion1_Run2_Grade
        # For now, let's just serialize it to make the CSV generation work.
        # The subtask mentions "per_bullet_run1_A, per_bullet_run2_A", which is very specific.
        # This implies knowing the criteria names AND that "A" is the field of interest.
        # This part is complex to generalize without knowing the exact structure of 'per_bullet_runs_aggregation'
        # and all possible criteria. A simpler, more general approach for now:
        flat_row['Per Bullet Runs Aggregation (JSON)'] = json.dumps(aggregation) if aggregation else ''
        
        flat_results.append(flat_row)

    df = pd.DataFrame(flat_results)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_rubric_name = "".join(c if c.isalnum() else "_" for c in selected_rubric_name)
    csv_filename = f"{filename_prefix}_{safe_rubric_name}_{timestamp}.csv"
    
    try:
        df.to_csv(csv_filename, index=False, encoding='utf-8-sig') # utf-8-sig for Excel compatibility
        logging.info(f"Results saved to {csv_filename}")
        return csv_filename
    except Exception as e:
        logging.error(f"Error saving CSV {csv_filename}: {e}")
        st.error(f"Could not save results to CSV: {e}")
        return ""

# --- Main Application ---
def main():
    st.set_page_config(page_title="NCEA CV Analysis Tool", layout="wide")
    st.title("NCEA CV Analysis & Misconception Tool")

    if not authenticate_user():
        st.info("Please login to use the application.")
        return

    # Initialize OpenRouter client (once per session after API key is available)
    if 'api_key' not in st.session_state:
        st.session_state.api_key = "" # Initialize if not present

    if 'api_key_valid' not in st.session_state:
        st.session_state.api_key_valid = None # None = unknown, True = valid, False = invalid

    if 'openrouter_client' not in st.session_state and st.session_state.api_key and st.session_state.api_key_valid is not False:
        try:
            st.session_state.openrouter_client = OpenRouter(api_key=st.session_state.api_key)
            # A simple test call or version check could be done here if API allows
            st.session_state.api_key_valid = True 
        except Exception as e:
            st.sidebar.error(f"Failed to initialize API client: {e}")
            st.session_state.api_key_valid = False
            # Do not proceed if client cannot be initialized

    # --- Sidebar Configuration ---
    with st.sidebar:
        st.header("Configuration")
        
        # API Key Input (moved to be always visible but disabled if key is 'valid')
        current_api_key = st.text_input(
            "OpenRouter API Key", 
            type="password", 
            key="api_key_input", # Use a different key from st.session_state.api_key
            value=st.session_state.api_key,
            help="Enter your OpenRouter API key.",
            disabled=st.session_state.api_key_valid is True 
        )
        if current_api_key != st.session_state.api_key:
            st.session_state.api_key = current_api_key
            st.session_state.api_key_valid = None # Reset validity on key change
            if 'openrouter_client' in st.session_state: del st.session_state['openrouter_client'] # Force re-init
            st.rerun()

        if st.session_state.api_key_valid is False and st.session_state.api_key:
            st.error("API Key is invalid or authentication failed. Please check and re-enter.")
        elif st.session_state.api_key_valid is True:
            st.success("API Key is valid.")
        
        # LLM Settings
        st.subheader("LLM Settings")
        # Retrieve from session state or set default
        st.session_state.grading_llm_model = st.selectbox(
            "Grading LLM Model", 
            options=["openai/gpt-3.5-turbo", "openai/gpt-4", "anthropic/claude-2", "google/gemini-pro"], 
            index=0, key="grading_model_select",
            help="Model used for grading individual CVs."
        )
        st.session_state.grading_temperature = st.slider(
            "Grading Temperature", 0.0, 1.0, 0.3, 0.05, key="grading_temp_slider",
            help="Lower values = more deterministic, higher = more creative."
        )
        st.session_state.analysis_llm_model = st.selectbox(
            "Analysis LLM Model", 
            options=["openai/gpt-4", "anthropic/claude-2", "openai/gpt-3.5-turbo", "google/gemini-pro"], 
            index=0, key="analysis_model_select",
            help="Model used for generating misconception analyses."
        )
        st.session_state.analysis_temperature = st.slider(
            "Analysis Temperature", 0.0, 1.0, 0.5, 0.05, key="analysis_temp_slider",
            help="Temperature for misconception analysis generation."
        )
        st.session_state.max_output_tokens = st.slider(
            "Max Output Tokens (Grading)", 500, 4000, 1500, 100, key="max_tokens_slider",
            help="Max tokens the grading LLM can output. Affects detail and cost."
        ) # This was in process_student_portfolio, now global

        st.session_state.num_grading_runs = st.number_input(
            "Number of Grading Runs per Portfolio", min_value=1, max_value=5, value=3, step=1,
            key="num_runs_input", help="Grade each portfolio multiple times for consistency check."
        )

        # Rubric Selection
        st.subheader("Rubric")
        rubrics_data = load_rubric_files() # Uses default "rubrics" dir
        if not rubrics_data:
            st.error("No rubrics loaded. Please add valid rubric JSON files to the 'rubrics' directory.")
            return # Stop execution if no rubrics
        
        # Use previously selected rubric if available, otherwise first one or None
        current_selected_rubric_name = st.session_state.get('selected_rubric_name', list(rubrics_data.keys())[0] if rubrics_data else None)
        
        selected_rubric_name = st.selectbox(
            "Select Rubric", options=list(rubrics_data.keys()), 
            key="rubric_select", index=list(rubrics_data.keys()).index(current_selected_rubric_name) if current_selected_rubric_name in rubrics_data else 0
        )
        st.session_state.selected_rubric_name = selected_rubric_name
        st.session_state.rubric_data = rubrics_data[selected_rubric_name]


        # File Input
        st.subheader("CV Input")
        upload_option = st.radio(
            "Choose CV source:", 
            ("Upload PDF files", "Process local directory (PDFs)", "Process ZIP file"), 
            key="cv_source_radio"
        )

        if 'student_files_to_process' not in st.session_state:
            st.session_state.student_files_to_process = []

        # Initialize temp_upload_dir in session state if not present
        if 'temp_upload_dir' not in st.session_state:
            st.session_state.temp_upload_dir = None


        if upload_option == "Upload PDF files":
            uploaded_files = st.file_uploader(
                "Upload one or more PDF files", type="pdf", 
                accept_multiple_files=True, key="pdf_file_uploader"
            )
            if uploaded_files:
                if st.session_state.temp_upload_dir is None or not Path(st.session_state.temp_upload_dir).exists():
                    st.session_state.temp_upload_dir = tempfile.mkdtemp(prefix=TEMP_UPLOAD_DIR_NAME)
                    logging.info(f"Created temp directory for uploads: {st.session_state.temp_upload_dir}")

                temp_dir_path = Path(st.session_state.temp_upload_dir)
                for uploaded_file in uploaded_files:
                    file_path = temp_dir_path / uploaded_file.name
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    if file_path not in st.session_state.student_files_to_process:
                         st.session_state.student_files_to_process.append(file_path)
                st.success(f"{len(uploaded_files)} file(s) uploaded and added to queue.")
        
        elif upload_option == "Process local directory (PDFs)":
            local_dir_path_str = st.text_input("Enter path to local directory:", key="local_dir_input", placeholder=f"e.g., {DEFAULT_CACHE_DIR}")
            if local_dir_path_str:
                local_dir_path = Path(local_dir_path_str)
                if local_dir_path.is_dir():
                    pdf_files = list(local_dir_path.glob("*.pdf"))
                    if pdf_files:
                        st.session_state.student_files_to_process.extend(pdf_files)
                        st.success(f"Added {len(pdf_files)} PDFs from '{local_dir_path_str}' to queue.")
                    else:
                        st.warning(f"No PDF files found in '{local_dir_path_str}'.")
                else:
                    st.error(f"Path '{local_dir_path_str}' is not a valid directory.")

        elif upload_option == "Process ZIP file":
            zip_file = st.file_uploader("Upload a ZIP file containing PDFs", type="zip", key="zip_uploader")
            if zip_file:
                if st.session_state.temp_upload_dir is None or not Path(st.session_state.temp_upload_dir).exists():
                    st.session_state.temp_upload_dir = tempfile.mkdtemp(prefix=TEMP_UPLOAD_DIR_NAME)
                
                zip_extract_base = Path(st.session_state.temp_upload_dir)
                # Create unique subfolder for this ZIP's contents
                unique_zip_folder_name = f"zip_{zip_file.name.split('.')[0]}_{os.urandom(4).hex()}"
                zip_extract_path = zip_extract_base / unique_zip_folder_name
                zip_extract_path.mkdir(parents=True, exist_ok=True)

                try:
                    with zipfile.ZipFile(zip_file, 'r') as zf:
                        zf.extractall(zip_extract_path)
                    extracted_pdfs = list(zip_extract_path.rglob("*.pdf")) # rglob for subdirs
                    if extracted_pdfs:
                        st.session_state.student_files_to_process.extend(extracted_pdfs)
                        st.success(f"Extracted {len(extracted_pdfs)} PDFs from '{zip_file.name}' to queue.")
                    else:
                        st.warning(f"No PDF files found in the ZIP '{zip_file.name}'.")
                except zipfile.BadZipFile:
                    st.error(f"Error: Uploaded file '{zip_file.name}' is not a valid ZIP file or is corrupted.")
                except Exception as e:
                    st.error(f"Error processing ZIP '{zip_file.name}': {e}")


        if st.button("Clear Queued Files", key="clear_queue_button"):
            st.session_state.student_files_to_process = []
            if st.session_state.temp_upload_dir and Path(st.session_state.temp_upload_dir).exists():
                try:
                    # shutil.rmtree(st.session_state.temp_upload_dir) # More robust for non-empty dirs
                    # For now, simple os.rmdir if it's empty or use Path.unlink for files
                    # This needs more robust cleanup logic for directories with files.
                    # Let's assume for now it mostly contains files or we handle cleanup elsewhere.
                    # A proper cleanup would iterate and delete.
                    logging.info(f"Cleared file queue. Temp dir {st.session_state.temp_upload_dir} may need manual cleanup if not empty or handle on session end.")
                    # For simplicity, just clear path. Actual deletion is complex with active files.
                    # st.session_state.temp_upload_dir = None 
                except Exception as e:
                    st.warning(f"Could not fully clear temporary upload directory: {e}")
            st.info("File queue cleared.")
            st.rerun()

        if st.session_state.student_files_to_process:
            st.write(f"Files in queue: {len(st.session_state.student_files_to_process)}")
            with st.expander("Show Queued Files"):
                for f_path in st.session_state.student_files_to_process:
                    st.caption(str(f_path))
    
    # --- Main Area for Processing and Results ---
    if not st.session_state.api_key or st.session_state.api_key_valid is not True :
        st.warning("Please enter a valid OpenRouter API Key in the sidebar to proceed.")
        return
    
    if 'openrouter_client' not in st.session_state:
        try:
            st.session_state.openrouter_client = OpenRouter(api_key=st.session_state.api_key)
            st.session_state.api_key_valid = True 
        except Exception as e:
            st.error(f"Failed to initialize API client with the provided key: {e}")
            st.session_state.api_key_valid = False
            return


    if st.button("Start Grading Process", key="start_grading_button", disabled=not st.session_state.student_files_to_process):
        if not st.session_state.student_files_to_process:
            st.error("No student portfolios loaded. Please upload or select files.")
            return

        # Retrieve settings from session_state (set by sidebar widgets)
        grading_model = st.session_state.grading_llm_model
        grading_temp = st.session_state.grading_temperature
        # max_tokens_grading = st.session_state.max_output_tokens # Already in process_student_portfolio
        num_runs = st.session_state.num_grading_runs
        
        llm_rubric_data = st.session_state.rubric_data
        openrouter_client_instance = st.session_state.openrouter_client


        progress_bar = st.progress(0)
        status_container = st.container()
        total_files = len(st.session_state.student_files_to_process)
        all_detailed_canonical_results = [] # Store the chosen result for each file

        for i, file_path_obj in enumerate(st.session_state.student_files_to_process):
            status_container.text(f"Processing {file_path_obj.name} ({i+1}/{total_files})...")
            
            # Extract text (once per file)
            student_cv_text, text_extract_warnings, _ = extract_text_from_pdf(str(file_path_obj))
            if text_extract_warnings:
                 for warn in text_extract_warnings: status_container.warning(f"Text extraction warning for {file_path_obj.name}: {warn}")

            grades_from_multiple_runs = []
            detailed_results_for_file_runs = []

            for run_num in range(num_runs):
                status_container.text(f"Grading {file_path_obj.name} - Run {run_num + 1}/{num_runs}")
                # process_student_portfolio is async
                individual_run_result = asyncio.run(process_student_portfolio(
                    student_text=student_cv_text, 
                    pdf_path=file_path_obj, 
                    llm_rubric_json=llm_rubric_data, 
                    api_key=st.session_state.api_key, # API key already set in client, but good to have if needed
                    model_name=grading_model, 
                    temperature=grading_temp,
                    openrouter_client=openrouter_client_instance
                ))
                detailed_results_for_file_runs.append(individual_run_result)
                grades_from_multiple_runs.append(individual_run_result['grade'])
            
            # Final Grade Logic
            valid_grades = [g for g in grades_from_multiple_runs if g not in ['Processing Error', 'Extraction Failed', 'Prompt Error', 'API Error', 'LLM Format Error', 'Ungraded']]
            final_assigned_grade = "Error In All Runs"
            grade_variability_info = "N/A"

            if valid_grades:
                grade_counts = Counter(valid_grades)
                final_assigned_grade = grade_counts.most_common(1)[0][0] # Mode
                if len(grade_counts) > 1:
                    grade_variability_info = f"Yes (Grades: {dict(grade_counts)})"
                else:
                    grade_variability_info = "No"
            elif grades_from_multiple_runs: # All runs resulted in errors
                final_assigned_grade = grades_from_multiple_runs[0] # Use the grade from the first error
                grade_variability_info = "All runs failed."

            # Canonical Result Selection (first successful, or first error if all failed)
            canonical_result = None
            first_successful_run = next((r for r in detailed_results_for_file_runs if r['error_stage'] is None), None)
            if first_successful_run:
                canonical_result = first_successful_run
            elif detailed_results_for_file_runs: # If all failed, pick the first one
                canonical_result = detailed_results_for_file_runs[0]
            
            if canonical_result:
                canonical_result_augmented = canonical_result.copy() # Avoid modifying original
                canonical_result_augmented['final_grade_assigned'] = final_assigned_grade
                # Placeholder for per_bullet_runs_aggregation; this is complex
                # It requires knowing all criteria from the rubric and how process_student_portfolio structures its output for each.
                # Assuming process_student_portfolio's 'failed_criteria' is what we might aggregate or list per run.
                # For now, just storing the list of grades for each run in this placeholder.
                canonical_result_augmented['per_bullet_runs_aggregation'] = {'grades_all_runs': grades_from_multiple_runs}
                canonical_result_augmented['grade_variability_info'] = grade_variability_info
                canonical_result_augmented['file_path'] = str(file_path_obj) # Store path for re-extraction
                all_detailed_canonical_results.append(canonical_result_augmented)
            
            progress_bar.progress((i + 1) / total_files)

        status_container.success("All portfolios processed!")
        st.session_state.cached_detailed_results_for_insights = all_detailed_canonical_results
        st.session_state.cached_insights_rubric_data = llm_rubric_data
        st.session_state.cached_analysis_llm_model = st.session_state.analysis_llm_model
        st.session_state.cached_analysis_temperature = st.session_state.analysis_temperature

        # Display summary table (optional, can be large)
        if all_detailed_canonical_results:
            st.subheader("Grading Summary")
            summary_df_data = [{
                "File": res.get('student_file'), 
                "Final Grade": res.get('final_grade_assigned'),
                "Variability": res.get('grade_variability_info', 'N/A'),
                "Error Stage (First/Canonical)": res.get('error_stage', "None")
            } for res in all_detailed_canonical_results]
            st.dataframe(pd.DataFrame(summary_df_data))

            csv_filename = save_detailed_results_to_csv(
                all_detailed_canonical_results, 
                "grading_details", 
                st.session_state.selected_rubric_name
            )
            if csv_filename:
                with open(csv_filename, "rb") as f:
                    st.download_button(
                        label="Download Detailed Results as CSV",
                        data=f,
                        file_name=Path(csv_filename).name,
                        mime="text/csv",
                        key="download_csv_button"
                    )
        st.session_state.student_files_to_process = [] # Clear queue after processing

    # --- Insights Generation Section ---
    st.divider()
    st.header("Misconception Analysis")

    if 'cached_detailed_results_for_insights' not in st.session_state:
        st.session_state.cached_detailed_results_for_insights = []

    if st.button("Generate Misconception Analysis", key="generate_insights_button", 
                  disabled=not st.session_state.cached_detailed_results_for_insights):
        
        insights_data = st.session_state.cached_detailed_results_for_insights
        rubric_for_insights = st.session_state.cached_insights_rubric_data
        analysis_model = st.session_state.cached_analysis_llm_model
        analysis_temp = st.session_state.cached_analysis_temperature # Ensure this is retrieved
        openrouter_client_instance = st.session_state.openrouter_client


        # Step 8: Generate failed_criteria_indepth if missing
        # This is a simplified version. A more robust one would update a DataFrame.
        # For now, modifying the list of dicts in place.
        updated_insights_data = []
        with st.spinner("Preparing data for insights (explaining failed criteria if needed)..."):
            for result_item in insights_data:
                if 'failed_criteria_indepth' not in result_item or not result_item['failed_criteria_indepth']:
                    if result_item.get('failed_criteria') and result_item.get('error_stage') is None : # Only if initial grading was successful and had failed criteria
                        # Re-extract text (or retrieve from a stored location if available)
                        # For now, assume file_path was stored in canonical_result_augmented
                        cv_text_for_expl, _, _ = extract_text_from_pdf(result_item['file_path'])
                        
                        # Simplified rubric text for explanation context
                        rubric_text_for_expl = f"Rubric: {rubric_for_insights.get('as_title', 'N/A')}\n"
                        # In a real scenario, you might pass more criteria details
                        
                        if cv_text_for_expl:
                            explanations = asyncio.run(explain_failed_criteria_with_llm(
                                failed_criteria=result_item['failed_criteria'],
                                cv_text=cv_text_for_expl,
                                rubric_text=rubric_text_for_expl, # Simplified rubric text
                                api_key=st.session_state.api_key, # Client already has key
                                model_name=analysis_model, # Use analysis model for this too
                                openrouter_client=openrouter_client_instance
                            ))
                            result_item['failed_criteria_indepth'] = explanations
                        else:
                             result_item['failed_criteria_indepth'] = [{"criterion": fc, "explanation": "Could not re-extract text for explanation."} for fc in result_item.get('failed_criteria',[])]
                    else: # No failed criteria or initial processing error
                         result_item['failed_criteria_indepth'] = []
                updated_insights_data.append(result_item)
            st.session_state.cached_detailed_results_for_insights = updated_insights_data # Update cache

        # TODO: Calculate common_misconceptions & grade_distribution from updated_insights_data
        # This is a placeholder for actual calculation logic based on results.
        # For now, passing empty structures.
        placeholder_grade_dist = Counter(res['final_grade_assigned'] for res in updated_insights_data if res.get('final_grade_assigned'))
        
        # Placeholder for common_misconceptions (this needs more complex aggregation)
        all_fc_explanations = []
        for res in updated_insights_data:
            if res.get('failed_criteria_indepth'):
                all_fc_explanations.extend(res['failed_criteria_indepth'])
        
        # Example: top 3 common criteria based on 'criterion' field in explanations
        if all_fc_explanations:
            fc_counts = Counter(expl.get('criterion') for expl in all_fc_explanations if isinstance(expl, dict) and expl.get('criterion'))
            placeholder_common_misconceptions = [
                {"criterion": crit, "percentage": (count / len(updated_insights_data)) * 100 if updated_insights_data else 0}
                for crit, count in fc_counts.most_common(5) # Top 5
            ]
            # Placeholder for common_misconceptions_by_level (even more complex)
            placeholder_common_misconceptions_by_level = {
                level: placeholder_common_misconceptions for level in GRADE_ORDER # Oversimplified
            }
        else:
            placeholder_common_misconceptions = []
            placeholder_common_misconceptions_by_level = {}


        with st.spinner("Generating Overall Misconception Analysis..."):
            overall_analysis_md = asyncio.run(run_overall_misconception_analysis(
                insights_df_source=updated_insights_data, 
                llm_rubric_json=rubric_for_insights,
                grade_distribution=dict(placeholder_grade_dist), 
                common_misconceptions=placeholder_common_misconceptions,
                api_key=st.session_state.api_key,
                model_name=analysis_model,
                openrouter_client=openrouter_client_instance
            ))
            st.markdown(overall_analysis_md)

        with st.spinner("Generating Per-Level Misconception Analysis..."):
            per_level_analysis_md = asyncio.run(run_per_level_misconception_analysis(
                insights_df_source=updated_insights_data,
                llm_rubric_json=rubric_for_insights,
                common_misconceptions_by_level=placeholder_common_misconceptions_by_level, # Placeholder
                api_key=st.session_state.api_key,
                model_name=analysis_model,
                openrouter_client=openrouter_client_instance
            ))
            st.markdown(per_level_analysis_md)

if __name__ == "__main__":
    # Setup dummy rubrics and cache for local testing if they don't exist
    Path(DEFAULT_RUBRICS_DIR).mkdir(exist_ok=True)
    Path(DEFAULT_CACHE_DIR).mkdir(exist_ok=True) 
    # Example dummy rubric (ensure it has as_code, as_title, gradingCriteria)
    dummy_rubric_path = Path(DEFAULT_RUBRICS_DIR) / "dummy_rubric.json"
    if not dummy_rubric_path.exists():
        with open(dummy_rubric_path, "w") as f:
            json.dump({
                "as_code": "99999", 
                "as_title": "Dummy Test Rubric",
                "gradingCriteria": [
                    {"levelName": "Basic Skill", "mainRequirement": "Demonstrates basic skill."}
                ]
            }, f)
    
    # Example dummy PDF in cache
    dummy_pdf_path = Path(DEFAULT_CACHE_DIR) / "dummy_cv.pdf"
    if not dummy_pdf_path.exists():
        try:
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text((50,72), "This is a dummy CV for testing.")
            doc.save(str(dummy_pdf_path))
            doc.close()
        except Exception as e:
            logging.warning(f"Could not create dummy PDF for testing: {e}")

    main()
