import os
import io
import json
import re
import warnings
import streamlit as st
import pandas as pd
from utils.text_extract import extract_text_from_pdf, extract_text_from_docx

# Silence pdfplumber cropbox warnings (optional)
warnings.filterwarnings("ignore", message="CropBox missing from /Page", module="pdfplumber")

# --- Streamlit Session State Initialization (MUST BE FIRST) ---
# We initialize variables here to persist data across reruns caused by slider/filter changes
if 'analysis_state' not in st.session_state:
    st.session_state.analysis_state = {
        'run': False,
        'results_df_raw': pd.DataFrame(), # Stores results *after* keyword matching but *before* final threshold status applied
        'docs_text': [],
        'last_run_files': set(),
        'last_run_standard': ""
    }

# --- Helpers ---
def project_path(*parts):
    """Return path relative to project root (where this file lives)."""
    base = os.path.dirname(__file__)
    return os.path.join(base, *parts)

def load_checklist(file_name):
    """Load checklist from JSON and normalize entries."""
    path = project_path("checklists", file_name)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except UnicodeDecodeError:
        with open(path, "r", encoding="cp1252", errors="ignore") as f:
            data = json.load(f)

    # Normalize entries: ensure keys exist and attach origin filename (standard)
    standard_label = file_name.replace("checklist_", "").replace(".json", "")
    normalized = []
    for entry in data:
        req = {
            "clause": entry.get("clause", "").strip(),
            "requirement": entry.get("requirement", "").strip(),
            "category": entry.get("category", "").strip() if entry.get("category") else "",
            "system": entry.get("system", "").strip() if entry.get("system") else standard_label,
            "origin_file": file_name
        }
        if req["requirement"]: # Only keep entries that have requirement text
            normalized.append(req)
    return normalized

def extract_keywords(req_text, max_keywords=8):
    """Extract meaningful keywords from requirement text."""
    if not req_text:
        return []
    
    # Expanded stop words
    stop_words = {
        "have", "has", "had", "you", "your", "the", "a", "an", "and", "or", "is", "are", "was", "were",
        "do", "does", "did", "can", "could", "will", "would", "shall", "should", "must", "may", "might",
        "this", "that", "these", "those", "what", "which", "who", "whom", "how", "when", "where", "why",
        "there", "their", "them", "then", "than", "thus", "therefore", "however", "although", "while",
        "with", "from", "for", "about", "against", "between", "into", "through", "during", "before",
        "after", "above", "below", "under", "again", "further", "once", "here", "both", "each", "few",
        "more", "most", "such", "only", "own", "same", "so", "than", "too", "very", "just", "now",
        "all", "mgt", "management", "system", "process", "document", "records", "shall", "be", "it",
        "need", "determine", "identify", "provide", "ensure", "monitor", "measure"
    }
    
    # Clean the text
    cleaned = re.sub(r'[^\w\s]', ' ', req_text.lower())
    
    # Tokenize and filter
    tokens = re.findall(r'\b[a-zA-Z]{4,}\b', cleaned) # Increased min length to 4
    
    # Filter out stop words and keep only meaningful tokens
    keywords = [token for token in tokens if token not in stop_words]
    
    # Remove duplicates while preserving order
    seen = set()
    unique_keywords = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            unique_keywords.append(kw)
    
    return unique_keywords[:max_keywords]

def analyze_checklist(checklist, docs_text, min_match_threshold=0.3, progress_callback=None):
    """
    Analyzes checklist requirements against document text.
    Returns results including raw match scores and evidence, which are then used
    to determine the final status based on the dynamic threshold.
    """
    results = []
    total = len(checklist)
    
    for i, req in enumerate(checklist):
        req_text = req.get("requirement", "")
        
        evidence = ""
        evidence_doc = ""
        best_match_score = 0 # Stored as percentage (0-100)
        
        if req_text:
            keywords = extract_keywords(req_text)
            
            if not keywords:
                best_match_score = 0
            else:
                best_match_text = ""
                best_match_doc = ""
                
                for d in docs_text:
                    txt = (d.get("text") or "").lower()
                    if not txt:
                        continue
                    
                    found_keywords = set()
                    for kw in keywords:
                        # Use word boundary matching
                        pattern = rf"\b{re.escape(kw)}\b"
                        if re.search(pattern, txt):
                            found_keywords.add(kw)
                    
                    match_percentage = (len(found_keywords) / len(keywords)) * 100 if keywords else 0
                    
                    if match_percentage > best_match_score:
                        best_match_score = match_percentage
                        
                        # Find a representative snippet using the keyword that provided the match
                        for kw in found_keywords:
                            pattern = rf"\b{re.escape(kw)}\b"
                            match = re.search(pattern, txt)
                            if match:
                                idx = match.start()
                                original = d.get("text") or ""
                                start = max(0, idx - 150)
                                end = min(len(original), idx + 150)
                                snippet = original[start:end].strip()
                                # Clean up the snippet
                                if start > 0:
                                    snippet = "..." + snippet
                                if end < len(original):
                                    snippet = snippet + "..."
                                best_match_text = snippet
                                best_match_doc = d.get("name", "")
                                break
                
                evidence = best_match_text
                evidence_doc = best_match_doc
        
        # NOTE: We do NOT apply the status here yet. We store the raw score.
        
        results.append({
            "standard": req.get("system", ""),
            "clause": req.get("clause", ""),
            "requirement": req_text,
            "match_score": best_match_score, # Store the raw score
            "evidence": evidence,
            "evidence_document": evidence_doc
        })

        if progress_callback:
            progress_callback(i + 1, total)

    # Now, dynamically apply the status based on the input threshold
    df = pd.DataFrame(results)
    df['status'] = df.apply(
        lambda row: calculate_status(row['match_score'], min_match_threshold), 
        axis=1
    )
    return df.to_dict('records') # Return as list of dicts for consistency

def calculate_status(match_score, threshold):
    """Applies the compliance logic based on a match score and threshold."""
    threshold_percentage = threshold * 100
    
    if match_score >= threshold_percentage:
        return "Compliant"
    elif match_score >= (threshold_percentage * 0.5):
        return "Partial"
    else:
        return "Not Found"

# --- Streamlit UI ---
st.set_page_config(page_title="AI Audit Reviewer", layout="wide", initial_sidebar_state="expanded")

# Add a sidebar for additional options
with st.sidebar:
    st.header("Settings")
    # Assign key to threshold so its change triggers rerun, but we don't re-extract docs
    min_match_threshold = st.slider("Minimum Match Threshold", 0.1, 1.0, 0.3, 0.1, key="match_threshold",
                                    help="Adjust how strict the matching should be (higher = more strict). This updates instantly after analysis.")
    
    st.header("About")
    st.info("This tool helps auditors check client documents against ISO standards requirements.")

# Main interface
st.title("📋 AI-Powered Audit Reviewer")
st.markdown("Upload client documents to check against compliance checklists for ISO standards.")

# File upload section
st.header("1. Upload Documents")
uploaded_files = st.file_uploader(
    # Updated text to reflect TXT support
    "Drag and drop client documents (PDF, DOCX, XLSX, TXT)",
    type=["pdf", "docx", "xlsx", "txt"], # Added 'txt'
    accept_multiple_files=True,
    help="Upload policies, procedures, records, and other relevant documents"
)

# Checklist selection
st.header("2. Select Audit Standard")
checklist_option = st.radio(
    "Choose which standard to check against:",
    ["ISO 9001 (Quality)", "ISO 14001 (Env.)", "ISO 45001 (OH&S)", "All Standards"],
    index=0,
    horizontal=False,
    key="selected_standard" # Key added to force re-run if changed
)

mapping = {
    "ISO 9001 (Quality)": ["checklist_9001.json"],
    "ISO 14001 (Env.)": ["checklist_14001.json"],
    "ISO 45001 (OH&S)": ["checklist_45001.json"],
    "All Standards": [
        "checklist_9001.json",
        "checklist_14001.json",
        "checklist_45001.json",
    ],
}

# --- Button Logic (Triggers Full Analysis and stores to Session State) ---

# Check if current file set/standard choice invalidates the existing analysis
current_file_names = {f.name for f in uploaded_files} if uploaded_files else set()
needs_reanalysis = (
    st.session_state.analysis_state['run'] and
    (current_file_names != st.session_state.analysis_state['last_run_files'] or 
     checklist_option != st.session_state.analysis_state['last_run_standard'])
)

if needs_reanalysis:
    st.warning("Document files or Standard selection have changed. Please click 'Analyze Documents' again.")
    # Reset analysis flag to hide old results display and force button click
    st.session_state.analysis_state['run'] = False

if uploaded_files:
    analyze_clicked = st.button("🚀 Analyze Documents", type="primary", use_container_width=True)
else:
    st.warning("Please upload documents first")
    analyze_clicked = False

# Status / debug area
status_area = st.empty()

if analyze_clicked:
    st.session_state.analysis_state['run'] = False # Reset flag while running

    # 1. Load checklist(s)
    checklist_files = mapping.get(checklist_option, [])
    checklist = []
    loaded_counts = {}
    for f in checklist_files:
        try:
            data = load_checklist(f)
            checklist.extend(data)
            loaded_counts[f] = len(data)
        except FileNotFoundError:
            st.error(f"Checklist JSON not found: checklists/{f}")
        except Exception as e:
            st.error(f"Failed to load checklists/{f}: {e}")

    if not checklist:
        st.error("No checklist items loaded.")
        st.stop()

    status_area.markdown(f"Loaded checklist items from: {', '.join(checklist_files)}")
    
    # 2. Extract text from uploaded files (The heavy step)
    docs_text = []
    status_area.info("Extracting text from uploaded documents...")
    
    # Use st.spinner for a nice loading experience
    with st.spinner("Processing documents..."):
        for f in uploaded_files:
            try:
                raw = f.read()
                file_name_lower = f.name.lower() # Store lowercased name

                if file_name_lower.endswith(".pdf"):
                    txt = extract_text_from_pdf(raw)
                elif file_name_lower.endswith(".docx"):
                    txt = extract_text_from_docx(raw)
                elif file_name_lower.endswith(".xlsx"):
                    try:
                        excel_data = pd.read_excel(io.BytesIO(raw), sheet_name=None)
                        txt = " ".join(
                            str(val)
                            for sheet in excel_data.values()
                            for row in sheet.fillna("").astype(str).values
                            for val in row
                        )
                    except Exception as e:
                        txt = ""
                        st.warning(f"Failed to read Excel {f.name}: {e}")
                elif file_name_lower.endswith(".txt"): # New: TXT file support
                    txt = raw.decode('utf-8', errors='ignore')
                else:
                    txt = ""
            except Exception as e:
                txt = ""
                st.warning(f"Failed to extract text from {f.name}: {e}")
            
            if txt:
                docs_text.append({"name": f.name, "text": txt})
            # st.write(f"✓ Extracted {len(txt)} characters from {f.name}") # Comment out for cleaner UI

    if not docs_text:
        st.error("No text could be extracted from any uploaded documents. Please check file formats.")
        st.stop()

    # 3. Run analysis (keyword matching and initial scoring)
    progress_bar = st.progress(0)
    progress_text = st.empty()

    def progress_callback_local(done, total):
        progress = int(done / total * 100)
        progress_bar.progress(min(progress, 100))
        progress_text.text(f"Analyzing requirement {done}/{total}...")
    
    status_area.info("Running keyword analysis...")
    
    # Analyze - passing the initial threshold but the result df contains the raw score
    raw_results = analyze_checklist(checklist, docs_text, min_match_threshold, progress_callback_local)
    
    # 4. Store all results and parameters in Session State
    if raw_results:
        st.session_state.analysis_state['results_df_raw'] = pd.DataFrame(raw_results)
        st.session_state.analysis_state['docs_text'] = docs_text
        st.session_state.analysis_state['last_run_files'] = current_file_names
        st.session_state.analysis_state['last_run_standard'] = checklist_option
        st.session_state.analysis_state['run'] = True
        status_area.success("✅ Analysis complete - results ready!")
    else:
        st.error("Analysis produced no results.")

    progress_bar.progress(100)
    progress_text.empty()
    st.rerun() # Rerun to hit the display block immediately

# --- Display Logic (Runs on every rerun if analysis_state['run'] is True) ---

if st.session_state.analysis_state['run']:
    st.header("3. Analysis Results")
    
    # 1. Retrieve the raw analysis data (which doesn't depend on the threshold)
    df_raw = st.session_state.analysis_state['results_df_raw'].copy()

    # 2. RE-APPLY status logic using the current slider value
    # This is the crucial part that ensures the status updates instantly when the slider moves
    with st.spinner("Recalculating status based on threshold..."):
        df_raw['status'] = df_raw.apply(
            lambda row: calculate_status(row['match_score'], min_match_threshold),
            axis=1
        )
    
    df = df_raw # The final DataFrame used for display and filtering
    
    if not df.empty:
        # Summary metrics
        compliant_count = int((df["status"] == "Compliant").sum())
        not_found_count = int((df["status"] == "Not Found").sum())
        partial_count = int((df["status"] == "Partial").sum())
        total_count = len(df)

        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Requirements", total_count)
        col2.metric("Compliant", compliant_count, f"{compliant_count/total_count*100:.1f}%" if total_count else "0.0%")
        col3.metric("Partial", partial_count, f"{partial_count/total_count*100:.1f}%" if total_count else "0.0%")
        col4.metric("Not Found", not_found_count, f"{not_found_count/total_count*100:.1f}%" if total_count else "0.0%")

        # Status distribution chart
        status_dist = df["status"].value_counts()
        st.bar_chart(status_dist)

        # Summary by standard (for multiple checklists)
        if checklist_option == "All Standards" and 'standard' in df.columns:
            st.markdown("### Summary by Standard")
            standard_summary = df.groupby('standard').agg({
                'status': ['count', 
                            lambda x: (x == 'Compliant').sum(),
                            lambda x: (x == 'Partial').sum(),
                            lambda x: (x == 'Not Found').sum()]
            }).round(0)
            standard_summary.columns = ['Total', 'Compliant', 'Partial', 'Not Found']
            st.dataframe(standard_summary, use_container_width=True)

        # Debug information
        with st.expander("Debug Information"):
            current_checklist_files = mapping.get(checklist_option, [])
            st.write(f"**Current Match Threshold:** {min_match_threshold} ({min_match_threshold * 100}%)")
            st.write(f"**Checklists Used (Stored):** {', '.join(current_checklist_files)}")
            st.write(f"**Total Documents Processed:** {len(st.session_state.analysis_state['docs_text'])}")


        # Display main results table
        st.markdown("### Full Results Table")
        display_cols = ["standard", "clause", "requirement", "status", "evidence", "evidence_document"]
        
        # Add color coding for status
        def color_status(val):
            if val == "Compliant":
                return "color: green; font-weight: bold;"
            elif val == "Partial":
                return "color: orange; font-weight: bold;"
            elif val == "Not Found":
                return "color: red; font-weight: bold;"
            return ""
        
        styled_df = df[display_cols].style.map(color_status, subset=['status'])
        st.dataframe(styled_df, use_container_width=True, height=600)

        # Filter options
        st.markdown("### Filter Results")
        # Now using three columns for better organization
        col1, col2, col3 = st.columns(3) 
        
        # Get available document names for filtering
        available_documents = sorted(df['evidence_document'].unique().tolist())
        # Ensure the empty string (for requirements with no evidence) is at the start if present
        if "" in available_documents:
            available_documents.remove("")
            available_documents.insert(0, "(No Evidence Document)")
        
        with col1:
            # Document Filter (NEW)
            filter_document = st.multiselect(
                "Filter by Document:",
                options=available_documents,
                default=available_documents,
                key="filter_document"
            )
            # Map "(No Evidence Document)" back to "" for filtering
            filter_document_for_df = ["" if name == "(No Evidence Document)" else name for name in filter_document]

        
        with col2:
            # Filters now apply to the re-calculated 'df' and do NOT re-run the analysis block
            filter_status = st.multiselect(
                "Filter by Status:",
                options=["Compliant", "Partial", "Not Found"],
                default=["Compliant", "Partial", "Not Found"],
                key="filter_status"
            )
        
        with col3:
            # Add standard filter for multiple checklists
            if checklist_option == "All Standards" and 'standard' in df.columns:
                available_standards = df['standard'].unique().tolist()
                filter_standard = st.multiselect(
                    "Filter by Standard:",
                    options=available_standards,
                    default=available_standards,
                    key="filter_standard"
                )
            else:
                filter_standard = None
        
        # Apply filters
        filtered_df = df.copy()
        
        if filter_status:
            filtered_df = filtered_df[filtered_df["status"].isin(filter_status)]
        
        if filter_standard:
            filtered_df = filtered_df[filtered_df["standard"].isin(filter_standard)]

        # Apply new document filter
        if filter_document_for_df:
            filtered_df = filtered_df[filtered_df["evidence_document"].isin(filter_document_for_df)]
        
        if not filtered_df.empty:
            # Display filtered results
            st.write(f"Showing {len(filtered_df)} of {len(df)} results")
            
            st.dataframe(filtered_df[display_cols].style.map(color_status, subset=['status']), 
                            use_container_width=True, height=400)
            
            # Download filtered results
            csv_filtered = filtered_df[display_cols].to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download filtered results CSV", csv_filtered, 
                                "filtered_compliance_results.csv", "text/csv")
        else:
            st.info("No results match the selected filters")

        # CSV download for all results
        csv = df[display_cols].to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download all results CSV", csv, 
                            "compliance_analysis_results.csv", "text/csv")
    
    status_area.empty() # Clear status area once display is ready
