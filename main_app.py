# main_app.py
import streamlit as st
import pandas as pd
import os
import io
import json
import re
import warnings
from utils.text_extract import extract_text_from_pdf, extract_text_from_docx

# Silence warnings
warnings.filterwarnings("ignore", message="CropBox missing from /Page", module="pdfplumber")

# --- Common Helper Functions ---
def project_path(*parts):
    base = os.path.dirname(__file__)
    return os.path.join(base, *parts)

def load_checklist(file_name):
    path = project_path("checklists", file_name)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except UnicodeDecodeError:
        with open(path, "r", encoding="cp1252", errors="ignore") as f:
            data = json.load(f)

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
        if req["requirement"]:
            normalized.append(req)
    return normalized

def extract_keywords(req_text, max_keywords=8):
    if not req_text:
        return []
    
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
    
    cleaned = re.sub(r'[^\w\s]', ' ', req_text.lower())
    tokens = re.findall(r'\b[a-zA-Z]{4,}\b', cleaned)
    keywords = [token for token in tokens if token not in stop_words]
    
    seen = set()
    unique_keywords = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            unique_keywords.append(kw)
    
    return unique_keywords[:max_keywords]

def calculate_status(match_score, threshold_percentage):
    """Applies the compliance logic based on a match score and threshold percentage."""
    if match_score >= threshold_percentage:
        return "Compliant"
    elif match_score >= (threshold_percentage * 0.5):
        return "Partial"
    else:
        return "Not Found"

# --- Session State Initialization ---
if 'analysis_state' not in st.session_state:
    st.session_state.analysis_state = {
        'run': False,
        'results_df_raw': pd.DataFrame(),
        'docs_text': [],
        'last_run_files': set(),
        'last_run_standards': [],
        'current_mode': None
    }

# --- Main App Configuration ---
st.set_page_config(
    page_title="ATLAS Audit Assistant", 
    layout="wide", 
    initial_sidebar_state="expanded",
    page_icon="📋"
)

# --- Sidebar Navigation ---
st.sidebar.title("🔍 ATLAS Audit Assistant")
st.sidebar.markdown("---")

# Mode selection
app_mode = st.sidebar.radio(
    "Select Analysis Mode:",
    ["Stage 1 Documentation Review", "Standard Checklists Analysis"],
    index=0  # Default to Stage 1 as landing page
)

st.sidebar.markdown("---")
st.sidebar.info("""
**About This Tool:**
- **Stage 1 Review**: Focused analysis for Stage 1 audit preparation
- **Standard Checklists**: Detailed analysis against specific ISO standards
""")

# --- Filter Functions ---
def setup_filters(df, mode):
    """Setup comprehensive filters for results"""
    st.subheader("🔍 Filter Results")
    
    # Create columns for filters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Status filter
        status_options = ["Compliant", "Partial", "Not Found"]
        selected_status = st.multiselect(
            "Filter by Status:",
            options=status_options,
            default=status_options,
            key=f"{mode}_status_filter"
        )
    
    with col2:
        # Document filter
        available_docs = sorted(df['evidence_document'].unique().tolist())
        if "" in available_docs:
            available_docs.remove("")
            available_docs.insert(0, "(No Evidence Document)")
        
        selected_docs = st.multiselect(
            "Filter by Document:",
            options=available_docs,
            default=available_docs,
            key=f"{mode}_doc_filter"
        )
        # Map back to empty string for filtering
        selected_docs_filter = ["" if doc == "(No Evidence Document)" else doc for doc in selected_docs]
    
    with col3:
        # Standard filter (only for multi-standard analysis)
        if len(df['standard'].unique()) > 1:
            standard_options = sorted(df['standard'].unique().tolist())
            selected_standards = st.multiselect(
                "Filter by Standard:",
                options=standard_options,
                default=standard_options,
                key=f"{mode}_standard_filter"
            )
        else:
            selected_standards = None
    
    with col4:
        # Critical items filter (only for Stage 1)
        if mode == "stage1" and 'is_critical' in df.columns:
            critical_options = ["All", "Critical Only", "Non-Critical Only"]
            selected_critical = st.selectbox(
                "Filter by Criticality:",
                options=critical_options,
                key=f"{mode}_critical_filter"
            )
        else:
            selected_critical = None
    
    # Apply filters
    filtered_df = df.copy()
    
    # Apply status filter
    if selected_status:
        filtered_df = filtered_df[filtered_df["status"].isin(selected_status)]
    
    # Apply document filter
    if selected_docs_filter:
        filtered_df = filtered_df[filtered_df["evidence_document"].isin(selected_docs_filter)]
    
    # Apply standard filter
    if selected_standards:
        filtered_df = filtered_df[filtered_df["standard"].isin(selected_standards)]
    
    # Apply critical filter
    if selected_critical == "Critical Only":
        filtered_df = filtered_df[filtered_df["is_critical"] == True]
    elif selected_critical == "Non-Critical Only":
        filtered_df = filtered_df[filtered_df["is_critical"] == False]
    
    return filtered_df

def display_filtered_results(filtered_df, original_df, mode):
    """Display filtered results with download options"""
    
    st.write(f"**Showing {len(filtered_df)} of {len(original_df)} requirements**")
    
    # Define display columns based on mode
    if mode == "stage1":
        display_cols = ["clause", "requirement", "status", "evidence", "evidence_document", "is_critical"]
        
        # Color coding functions
        def color_status(val):
            if val == "Compliant": return "color: green; font-weight: bold;"
            elif val == "Partial": return "color: orange; font-weight: bold;"
            elif val == "Not Found": return "color: red; font-weight: bold;"
            return ""
        
        def color_critical(val):
            if val == True: return "background-color: #fff3cd; font-weight: bold;"
            return ""
        
        styled_df = filtered_df[display_cols].style.map(color_status, subset=['status']).map(color_critical, subset=['is_critical'])
        
    else:  # standard mode
        display_cols = ["standard", "clause", "requirement", "status", "evidence", "evidence_document"]
        
        def color_status(val):
            if val == "Compliant": return "color: green; font-weight: bold;"
            elif val == "Partial": return "color: orange; font-weight: bold;"
            elif val == "Not Found": return "color: red; font-weight: bold;"
            return ""
        
        styled_df = filtered_df[display_cols].style.map(color_status, subset=['status'])
    
    # Display filtered table
    st.dataframe(styled_df, use_container_width=True, height=400)
    
    # Download buttons for filtered results
    st.subheader("📥 Export Filtered Results")
    col1, col2 = st.columns(2)
    
    with col1:
        # CSV download
        csv_data = filtered_df[display_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Filtered CSV",
            csv_data,
            f"{mode}_filtered_results.csv",
            "text/csv",
            key=f"{mode}_filtered_csv"
        )
    
    with col2:
        # Summary download
        if mode == "stage1":
            critical_count = (filtered_df['is_critical'] == True).sum() if 'is_critical' in filtered_df.columns else 0
            critical_compliant = ((filtered_df['is_critical'] == True) & (filtered_df['status'] == 'Compliant')).sum() if 'is_critical' in filtered_df.columns else 0
            
            summary_text = f"""
Filtered Results Summary - {mode.upper()}
========================================

Total Requirements: {len(filtered_df)}
Compliant: {(filtered_df['status'] == 'Compliant').sum()} ({(filtered_df['status'] == 'Compliant').sum()/len(filtered_df)*100:.1f}%)
Partial: {(filtered_df['status'] == 'Partial').sum()} ({(filtered_df['status'] == 'Partial').sum()/len(filtered_df)*100:.1f}%)
Not Found: {(filtered_df['status'] == 'Not Found').sum()} ({(filtered_df['status'] == 'Not Found').sum()/len(filtered_df)*100:.1f}%)

Critical Requirements: {critical_count}
Critical Compliant: {critical_compliant} ({critical_compliant/critical_count*100:.1f}% if critical_count > 0 else 0)

Filters Applied:
- Status: {', '.join(filtered_df['status'].unique())}
- Documents: {len(filtered_df['evidence_document'].unique())} documents
- Standards: {', '.join(filtered_df['standard'].unique()) if 'standard' in filtered_df.columns else 'Single standard'}

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        else:
            summary_text = f"""
Filtered Results Summary - STANDARD CHECKLISTS
==============================================

Total Requirements: {len(filtered_df)}
Compliant: {(filtered_df['status'] == 'Compliant').sum()} ({(filtered_df['status'] == 'Compliant').sum()/len(filtered_df)*100:.1f}%)
Partial: {(filtered_df['status'] == 'Partial').sum()} ({(filtered_df['status'] == 'Partial').sum()/len(filtered_df)*100:.1f}%)
Not Found: {(filtered_df['status'] == 'Not Found').sum()} ({(filtered_df['status'] == 'Not Found').sum()/len(filtered_df)*100:.1f}%)

Standards: {', '.join(filtered_df['standard'].unique())}
Documents: {len(filtered_df['evidence_document'].unique())} documents

Filters Applied:
- Status: {', '.join(filtered_df['status'].unique())}
- Standards: {', '.join(filtered_df['standard'].unique())}

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        st.download_button(
            "Download Filtered Summary",
            summary_text,
            f"{mode}_filtered_summary.txt",
            "text/plain",
            key=f"{mode}_filtered_summary"
        )

# --- Stage 1 Analysis Function ---
def run_stage1_analysis():
    st.title("📋 Stage 1 Documentation Review Analyzer")
    st.markdown("""
    **Specialized tool for Stage 1 audit preparation**  
    Focused on the critical documentation required for Stage 1 readiness assessment.
    This tool analyzes your documentation against the integrated Stage 1 checklist.
    """)
    
    # File upload
    st.header("1. Upload Stage 1 Documentation")
    uploaded_files = st.file_uploader(
        "Upload your organization's documentation for Stage 1 review:",
        type=["pdf", "docx", "xlsx", "txt"],
        accept_multiple_files=True,
        help="Upload policies, procedures, manuals, records, and other relevant documents",
        key="stage1_upload"
    )
    
    # Stage 1 specific settings
    with st.sidebar:
        if app_mode == "Stage 1 Documentation Review":
            st.header("🔧 Analysis Settings")
            
            # Percentage-based threshold with layman descriptions
            threshold_percentage = st.slider(
                "Match Confidence Level",
                min_value=10,
                max_value=100,
                value=30,
                step=10,
                help="""
                **How strict should the analysis be?**
                
                🟢 **LOW (10-30%)**: More lenient - finds potential matches
                🟡 **MEDIUM (40-60%)**: Balanced approach  
                🔴 **HIGH (70-100%)**: Very strict - only strong matches
                
                *Lower values catch more potential issues, higher values reduce false positives*
                """,
                key="stage1_threshold"
            )
            
            # Show current threshold description
            if threshold_percentage <= 30:
                st.success(f"**Current: LOW ({threshold_percentage}%)** - Broad search for potential compliance")
            elif threshold_percentage <= 60:
                st.warning(f"**Current: MEDIUM ({threshold_percentage}%)** - Balanced approach")
            else:
                st.error(f"**Current: HIGH ({threshold_percentage}%)** - Strict matching only")
            
            st.header("🎯 Stage 1 Focus Areas")
            st.info("""
            **Critical Documentation:**
            - Scope definition
            - Policy establishment  
            - Risk assessment processes
            - Environmental aspects
            - Hazard identification
            - Compliance obligations
            - Objectives planning
            - Internal audit program
            - Management review
            """)

    # Analysis button for Stage 1
    if uploaded_files and st.button("🚀 Analyze Stage 1 Readiness", type="primary", use_container_width=True):
        with st.spinner("Running Stage 1 analysis..."):
            # Load Stage 1 checklist
            try:
                checklist = load_checklist("checklist_stage1.json")
                # Add Stage 1 specific metadata
                for item in checklist:
                    item['analysis_type'] = "Stage 1 Documentation Review"
                    item['stage'] = 'Stage 1'
            except FileNotFoundError:
                st.error("Stage 1 checklist not found. Please ensure checklist_stage1.json exists in the checklists folder.")
                return
            
            # Extract text from documents
            docs_text = []
            for f in uploaded_files:
                try:
                    raw = f.read()
                    file_name_lower = f.name.lower()

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
                    elif file_name_lower.endswith(".txt"):
                        txt = raw.decode('utf-8', errors='ignore')
                    else:
                        txt = ""
                except Exception as e:
                    txt = ""
                    st.warning(f"Failed to extract text from {f.name}: {e}")
                
                if txt:
                    docs_text.append({"name": f.name, "text": txt})
            
            if not docs_text:
                st.error("No text could be extracted from any uploaded documents. Please check file formats.")
                return

            # Run analysis
            progress_bar = st.progress(0)
            results = []
            
            for i, req in enumerate(checklist):
                req_text = req.get("requirement", "")
                best_match_score = 0
                evidence = ""
                evidence_doc = ""
                
                if req_text:
                    keywords = extract_keywords(req_text)
                    
                    if keywords:
                        for d in docs_text:
                            txt = (d.get("text") or "").lower()
                            if not txt:
                                continue
                            
                            found_keywords = set()
                            for kw in keywords:
                                pattern = rf"\b{re.escape(kw)}\b"
                                if re.search(pattern, txt):
                                    found_keywords.add(kw)
                            
                            match_percentage = (len(found_keywords) / len(keywords)) * 100 if keywords else 0
                            
                            if match_percentage > best_match_score:
                                best_match_score = match_percentage
                                # Simplified evidence extraction
                                for kw in found_keywords:
                                    pattern = rf"\b{re.escape(kw)}\b"
                                    match = re.search(pattern, txt)
                                    if match:
                                        idx = match.start()
                                        original = d.get("text") or ""
                                        start = max(0, idx - 150)
                                        end = min(len(original), idx + 150)
                                        snippet = original[start:end].strip()
                                        if start > 0:
                                            snippet = "..." + snippet
                                        if end < len(original):
                                            snippet = snippet + "..."
                                        evidence = snippet
                                        evidence_doc = d.get("name", "")
                                        break
                
                results.append({
                    "standard": req.get("system", ""),
                    "clause": req.get("clause", ""),
                    "requirement": req_text,
                    "match_score": best_match_score,
                    "evidence": evidence,
                    "evidence_document": evidence_doc,
                    "status": calculate_status(best_match_score, threshold_percentage)
                })
                
                progress_bar.progress((i + 1) / len(checklist))
            
            df = pd.DataFrame(results)
            
            # Calculate Stage 1 readiness
            critical_elements = [
                "internal audit", "management review", "scope", 
                "policy", "risks and opportunities", "environmental aspects",
                "hazards", "compliance obligations", "objectives"
            ]
            
            df['is_critical'] = df['requirement'].apply(
                lambda x: any(keyword in x.lower() for keyword in critical_elements)
            )
            
            critical_df = df[df['is_critical']]
            if len(critical_df) > 0:
                critical_compliant = (critical_df['status'] == 'Compliant').sum()
                stage1_score = (critical_compliant / len(critical_df)) * 100
            else:
                stage1_score = 0
            
            # Store results in session state
            st.session_state.analysis_state['results_df_raw'] = df
            st.session_state.analysis_state['run'] = True
            st.session_state.analysis_state['current_mode'] = "stage1"
            
            # Display results
            st.header("2. Stage 1 Readiness Assessment")
            
            # Show current threshold setting
            st.info(f"**Analysis Settings**: Match Confidence Level = {threshold_percentage}%")
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Stage 1 Readiness", f"{stage1_score:.1f}%")
            col2.metric("Critical Requirements", f"{critical_compliant}/{len(critical_df)}")
            col3.metric("Overall Compliance", f"{(df['status'] == 'Compliant').sum()}/{len(df)}")
            col4.metric("Compliance Rate", f"{(df['status'] == 'Compliant').sum()/len(df)*100:.1f}%" if len(df) > 0 else "0%")
            
            # Recommendation
            if stage1_score >= 80:
                st.success("✅ **READY for Stage 2 Audit** - Strong documentation foundation")
            elif stage1_score >= 60:
                st.warning("⚠️ **CONDITIONALLY READY** - Address minor gaps before Stage 2")
            else:
                st.error("❌ **NOT READY** - Significant documentation gaps need addressing")
            
            # Display full table first
            st.header("3. Detailed Analysis Results")
            display_cols = ["clause", "requirement", "status", "evidence", "evidence_document", "is_critical"]
            
            def color_status(val):
                if val == "Compliant": return "color: green; font-weight: bold;"
                elif val == "Partial": return "color: orange; font-weight: bold;"
                elif val == "Not Found": return "color: red; font-weight: bold;"
                return ""
            
            def color_critical(val):
                if val == True: return "background-color: #fff3cd; font-weight: bold;"
                return ""
            
            styled_df = df[display_cols].style.map(color_status, subset=['status']).map(color_critical, subset=['is_critical'])
            st.dataframe(styled_df, use_container_width=True, height=600)
            
            # Add filters section
            filtered_df = setup_filters(df, "stage1")
            
            if not filtered_df.empty:
                display_filtered_results(filtered_df, df, "stage1")
            else:
                st.info("No results match the selected filters")
            
            # Download full results
            st.header("4. Export Full Results")
            col1, col2 = st.columns(2)
            
            with col1:
                csv_full = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Full Analysis Results",
                    csv_full,
                    "stage1_full_analysis_results.csv",
                    "text/csv",
                    key="stage1_full_csv"
                )
            
            with col2:
                executive_summary = f"""
Stage 1 Readiness Assessment Report
===================================

Analysis Settings:
- Match Confidence Level: {threshold_percentage}%

Overall Readiness Score: {stage1_score:.1f}%
Recommendation: {"READY for Stage 2 Audit" if stage1_score >= 80 else "CONDITIONALLY READY" if stage1_score >= 60 else "NOT READY"}

Summary Metrics:
- Total Requirements: {len(df)}
- Compliant: {(df['status'] == 'Compliant').sum()} ({(df['status'] == 'Compliant').sum()/len(df)*100:.1f}%)
- Partial: {(df['status'] == 'Partial').sum()} ({(df['status'] == 'Partial').sum()/len(df)*100:.1f}%)
- Not Found: {(df['status'] == 'Not Found').sum()} ({(df['status'] == 'Not Found').sum()/len(df)*100:.1f}%)

Critical Requirements:
- Total Critical: {len(critical_df)}
- Critical Compliant: {critical_compliant} ({critical_compliant/len(critical_df)*100:.1f}%)

Analysis completed on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
                """
                
                st.download_button(
                    "📄 Download Executive Summary",
                    executive_summary,
                    "stage1_executive_summary.txt",
                    "text/plain",
                    key="stage1_summary"
                )

    elif not uploaded_files:
        st.info("👆 Please upload your organization's documentation to begin the Stage 1 analysis.")

# --- Standard Checklists Analysis Function ---
def run_standard_analysis():
    st.title("📊 Standard Checklists Analysis")
    st.markdown("""
    **Comprehensive analysis against specific ISO standards**  
    Analyze documentation against individual or combined ISO standard requirements.
    """)
    
    # File upload
    st.header("1. Upload Documents")
    uploaded_files = st.file_uploader(
        "Drag and drop client documents (PDF, DOCX, XLSX, TXT)",
        type=["pdf", "docx", "xlsx", "txt"],
        accept_multiple_files=True,
        help="Upload policies, procedures, records, and other relevant documents",
        key="standard_upload"
    )
    
    # FIXED: 2-Standard Combinations Support
    st.header("2. Select Standards for Analysis")
    selected_standards = st.multiselect(
        "Choose one or multiple standards to analyze against:",
        options=[
            "ISO 9001 (Quality)", 
            "ISO 14001 (Environmental)", 
            "ISO 45001 (OH&S)"
        ],
        default=["ISO 9001 (Quality)"],
        help="Select any combination of standards needed"
    )
    
    # Settings
    with st.sidebar:
        if app_mode == "Standard Checklists Analysis":
            st.header("🔧 Analysis Settings")
            
            # Percentage-based threshold with layman descriptions
            threshold_percentage = st.slider(
                "Match Confidence Level",
                min_value=10,
                max_value=100,
                value=30,
                step=10,
                help="""
                **How strict should the analysis be?**
                
                🟢 **LOW (10-30%)**: More lenient - finds potential matches
                🟡 **MEDIUM (40-60%)**: Balanced approach  
                🔴 **HIGH (70-100%)**: Very strict - only strong matches
                
                *Lower values catch more potential issues, higher values reduce false positives*
                """,
                key="standard_threshold"
            )
            
            # Show current threshold description
            if threshold_percentage <= 30:
                st.success(f"**Current: LOW ({threshold_percentage}%)** - Broad search for potential compliance")
            elif threshold_percentage <= 60:
                st.warning(f"**Current: MEDIUM ({threshold_percentage}%)** - Balanced approach")
            else:
                st.error(f"**Current: HIGH ({threshold_percentage}%)** - Strict matching only")

    # Analysis logic
    if uploaded_files and selected_standards:
        if st.button("🚀 Analyze Documents", type="primary", use_container_width=True):
            # Load selected checklists
            mapping = {
                "ISO 9001 (Quality)": "checklist_9001.json",
                "ISO 14001 (Environmental)": "checklist_14001.json", 
                "ISO 45001 (OH&S)": "checklist_45001.json"
            }
            
            checklist_files = [mapping[standard] for standard in selected_standards]
            checklist = []
            
            for f in checklist_files:
                try:
                    data = load_checklist(f)
                    checklist.extend(data)
                except FileNotFoundError:
                    st.error(f"Checklist JSON not found: checklists/{f}")
                    return
            
            if not checklist:
                st.error("No checklist items loaded.")
                return
            
            # Extract text from documents
            docs_text = []
            for f in uploaded_files:
                try:
                    raw = f.read()
                    file_name_lower = f.name.lower()

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
                    elif file_name_lower.endswith(".txt"):
                        txt = raw.decode('utf-8', errors='ignore')
                    else:
                        txt = ""
                except Exception as e:
                    txt = ""
                    st.warning(f"Failed to extract text from {f.name}: {e}")
                
                if txt:
                    docs_text.append({"name": f.name, "text": txt})
            
            if not docs_text:
                st.error("No text could be extracted from any uploaded documents. Please check file formats.")
                return

            # Run analysis
            progress_bar = st.progress(0)
            results = []
            
            for i, req in enumerate(checklist):
                req_text = req.get("requirement", "")
                best_match_score = 0
                evidence = ""
                evidence_doc = ""
                
                if req_text:
                    keywords = extract_keywords(req_text)
                    
                    if keywords:
                        for d in docs_text:
                            txt = (d.get("text") or "").lower()
                            if not txt:
                                continue
                            
                            found_keywords = set()
                            for kw in keywords:
                                pattern = rf"\b{re.escape(kw)}\b"
                                if re.search(pattern, txt):
                                    found_keywords.add(kw)
                            
                            match_percentage = (len(found_keywords) / len(keywords)) * 100 if keywords else 0
                            
                            if match_percentage > best_match_score:
                                best_match_score = match_percentage
                                # Evidence extraction
                                for kw in found_keywords:
                                    pattern = rf"\b{re.escape(kw)}\b"
                                    match = re.search(pattern, txt)
                                    if match:
                                        idx = match.start()
                                        original = d.get("text") or ""
                                        start = max(0, idx - 150)
                                        end = min(len(original), idx + 150)
                                        snippet = original[start:end].strip()
                                        if start > 0:
                                            snippet = "..." + snippet
                                        if end < len(original):
                                            snippet = snippet + "..."
                                        evidence = snippet
                                        evidence_doc = d.get("name", "")
                                        break
                
                results.append({
                    "standard": req.get("system", ""),
                    "clause": req.get("clause", ""),
                    "requirement": req_text,
                    "match_score": best_match_score,
                    "evidence": evidence,
                    "evidence_document": evidence_doc,
                    "status": calculate_status(best_match_score, threshold_percentage)
                })
                
                progress_bar.progress((i + 1) / len(checklist))
            
            df = pd.DataFrame(results)
            
            # Store results in session state
            st.session_state.analysis_state['results_df_raw'] = df
            st.session_state.analysis_state['run'] = True
            st.session_state.analysis_state['current_mode'] = "standard"
            
            # Display results
            st.header("3. Analysis Results")
            
            # Show current threshold setting
            st.info(f"**Analysis Settings**: Match Confidence Level = {threshold_percentage}%")
            
            # Summary metrics
            compliant_count = int((df["status"] == "Compliant").sum())
            not_found_count = int((df["status"] == "Not Found").sum())
            partial_count = int((df["status"] == "Partial").sum())
            total_count = len(df)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Requirements", total_count)
            col2.metric("Compliant", compliant_count, f"{compliant_count/total_count*100:.1f}%" if total_count else "0.0%")
            col3.metric("Partial", partial_count, f"{partial_count/total_count*100:.1f}%" if total_count else "0.0%")
            col4.metric("Not Found", not_found_count, f"{not_found_count/total_count*100:.1f}%" if total_count else "0.0%")

            # Display full table first
            st.header("4. Detailed Results")
            display_cols = ["standard", "clause", "requirement", "status", "evidence", "evidence_document"]
            
            def color_status(val):
                if val == "Compliant": return "color: green; font-weight: bold;"
                elif val == "Partial": return "color: orange; font-weight: bold;"
                elif val == "Not Found": return "color: red; font-weight: bold;"
                return ""
            
            styled_df = df[display_cols].style.map(color_status, subset=['status'])
            st.dataframe(styled_df, use_container_width=True, height=600)
            
            # Add filters section
            filtered_df = setup_filters(df, "standard")
            
            if not filtered_df.empty:
                display_filtered_results(filtered_df, df, "standard")
            else:
                st.info("No results match the selected filters")
            
            # Download full results
            st.header("5. Export Full Results")
            csv_full = df[display_cols].to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Full Analysis Results",
                csv_full,
                "standard_analysis_results.csv",
                "text/csv",
                key="standard_full_csv"
            )
    
    elif not uploaded_files:
        st.warning("Please upload documents first")
    elif not selected_standards:
        st.warning("Please select at least one standard to analyze against")

# --- Main App Router ---
def main():
    if app_mode == "Stage 1 Documentation Review":
        run_stage1_analysis()
    else:
        run_standard_analysis()

if __name__ == "__main__":
    main()
