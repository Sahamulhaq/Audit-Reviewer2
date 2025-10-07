import streamlit as st
import pandas as pd
import os
import io
import json
import re
import warnings
from utils.text_extract import extract_text_from_pdf, extract_text_from_docx

# Silence pdfplumber cropbox warnings (optional)
warnings.filterwarnings("ignore", message="CropBox missing from /Page", module="pdfplumber")

# --- Helper Functions (copied from app.py) ---
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

def analyze_checklist_stage1(checklist, docs_text, min_match_threshold=0.3, progress_callback=None):
    """
    Stage 1 specific analysis - analyzes checklist requirements against document text.
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
        
        results.append({
            "standard": req.get("system", ""),
            "clause": req.get("clause", ""),
            "requirement": req_text,
            "match_score": best_match_score,
            "evidence": evidence,
            "evidence_document": evidence_doc
        })

        if progress_callback:
            progress_callback(i + 1, total)

    # Apply status based on threshold
    df = pd.DataFrame(results)
    df['status'] = df.apply(
        lambda row: calculate_status(row['match_score'], min_match_threshold), 
        axis=1
    )
    return df

def calculate_status(match_score, threshold):
    """Applies the compliance logic based on a match score and threshold."""
    threshold_percentage = threshold * 100
    
    if match_score >= threshold_percentage:
        return "Compliant"
    elif match_score >= (threshold_percentage * 0.5):
        return "Partial"
    else:
        return "Not Found"

# --- Stage 1 Analyzer Class ---
class Stage1Analyzer:
    def __init__(self):
        self.checklist_file = "checklist_stage1.json"
        self.analysis_type = "Stage 1 Documentation Review"
    
    def load_stage1_checklist(self):
        """Load the dedicated Stage 1 checklist"""
        try:
            checklist = load_checklist(self.checklist_file)
            # Add Stage 1 specific metadata
            for item in checklist:
                item['analysis_type'] = self.analysis_type
                item['stage'] = 'Stage 1'
            return checklist
        except FileNotFoundError:
            st.error(f"Stage 1 checklist not found: {self.checklist_file}")
            st.info("Please run parse_checklists.py first to generate the Stage 1 checklist JSON file.")
            return []
    
    def extract_documents_text(self, uploaded_files):
        """Extract text from uploaded documents"""
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
        
        return docs_text
    
    def run_stage1_analysis(self, uploaded_files, min_match_threshold=0.3):
        """Run Stage 1 specific analysis"""
        # Load Stage 1 checklist
        checklist = self.load_stage1_checklist()
        if not checklist:
            return None
            
        # Extract text from documents
        docs_text = self.extract_documents_text(uploaded_files)
        
        if not docs_text:
            st.error("No text could be extracted from any uploaded documents. Please check file formats.")
            return None

        # Run analysis
        progress_bar = st.progress(0)
        progress_text = st.empty()

        def progress_callback(done, total):
            progress = int(done / total * 100)
            progress_bar.progress(min(progress, 100))
            progress_text.text(f"Analyzing Stage 1 requirement {done}/{total}...")
        
        results_df = analyze_checklist_stage1(checklist, docs_text, min_match_threshold, progress_callback)
        
        progress_bar.progress(100)
        progress_text.empty()
        
        # Add Stage 1 specific scoring
        df = self.calculate_stage1_scores(results_df)
        return df
    
    def calculate_stage1_scores(self, results_df):
        """Calculate Stage 1 specific compliance scores"""
        df = results_df.copy()
        
        # Define critical elements for Stage 1
        critical_elements = [
            "internal audit", "management review", "scope", 
            "policy", "risks and opportunities", "environmental aspects",
            "hazards", "compliance obligations", "objectives"
        ]
        
        # Tag critical requirements
        df['is_critical'] = df['requirement'].apply(
            lambda x: any(keyword in x.lower() for keyword in critical_elements)
        )
        
        # Calculate Stage 1 readiness score
        critical_df = df[df['is_critical']]
        if len(critical_df) > 0:
            critical_compliant = (critical_df['status'] == 'Compliant').sum()
            stage1_score = (critical_compliant / len(critical_df)) * 100
        else:
            stage1_score = 0
        
        df['stage1_readiness_score'] = stage1_score
        return df
    
    def generate_stage1_report(self, results_df):
        """Generate Stage 1 specific audit report"""
        critical_df = results_df[results_df['is_critical']]
        
        report = {
            'analysis_type': self.analysis_type,
            'total_requirements': len(results_df),
            'compliant_count': (results_df['status'] == 'Compliant').sum(),
            'partial_count': (results_df['status'] == 'Partial').sum(),
            'not_found_count': (results_df['status'] == 'Not Found').sum(),
            'critical_compliant': (critical_df['status'] == 'Compliant').sum(),
            'total_critical': len(critical_df),
            'readiness_score': results_df['stage1_readiness_score'].iloc[0] if len(results_df) > 0 else 0,
            'recommendation': self.get_stage1_recommendation(results_df)
        }
        return report
    
    def get_stage1_recommendation(self, results_df):
        """Generate Stage 1 specific recommendation"""
        readiness_score = results_df['stage1_readiness_score'].iloc[0] if len(results_df) > 0 else 0
        
        if readiness_score >= 80:
            return "✅ READY for Stage 2 Audit - Strong documentation foundation"
        elif readiness_score >= 60:
            return "⚠️ CONDITIONALLY READY - Address minor gaps before Stage 2"
        else:
            return "❌ NOT READY - Significant documentation gaps need addressing"

def stage1_main():
    """Dedicated Stage 1 analysis interface"""
    st.set_page_config(page_title="Stage 1 Audit Analyzer", layout="wide", initial_sidebar_state="expanded")
    
    st.title("📋 Stage 1 Documentation Review Analyzer")
    st.markdown("""
    **Specialized tool for Stage 1 audit preparation**  
    Focused on the critical documentation required for Stage 1 readiness assessment.
    This tool analyzes your documentation against the integrated Stage 1 checklist.
    """)
    
    # Initialize analyzer
    analyzer = Stage1Analyzer()
    
    # File upload
    st.header("1. Upload Stage 1 Documentation")
    uploaded_files = st.file_uploader(
        "Upload your organization's documentation for Stage 1 review:",
        type=["pdf", "docx", "xlsx", "txt"],
        accept_multiple_files=True,
        help="Upload policies, procedures, manuals, records, and other relevant documents"
    )
    
    # Stage 1 specific settings
    with st.sidebar:
        st.header("Stage 1 Settings")
        min_match_threshold = st.slider(
            "Match Threshold", 0.1, 1.0, 0.3, 0.1,
            help="Adjust how strict the requirement matching should be (higher = more strict)"
        )
        
        st.header("Stage 1 Focus Areas")
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
        
        st.header("About")
        st.caption("This specialized tool focuses specifically on Stage 1 audit requirements for integrated management systems.")

    # Analysis button
    if uploaded_files and st.button("🚀 Analyze Stage 1 Readiness", type="primary", use_container_width=True):
        with st.spinner("Running Stage 1 analysis..."):
            results = analyzer.run_stage1_analysis(uploaded_files, min_match_threshold)
            
            if results is not None:
                # Display Stage 1 specific results
                report = analyzer.generate_stage1_report(results)
                
                st.header("2. Stage 1 Readiness Assessment")
                
                # Stage 1 readiness metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Stage 1 Readiness", f"{report['readiness_score']:.1f}%")
                col2.metric("Critical Requirements", 
                           f"{report['critical_compliant']}/{report['total_critical']}")
                col3.metric("Overall Compliance", 
                           f"{report['compliant_count']}/{report['total_requirements']}")
                col4.metric("Compliance Rate", 
                           f"{(report['compliant_count']/report['total_requirements']*100):.1f}%" if report['total_requirements'] > 0 else "0%")
                
                # Recommendation with emphasis
                st.subheader("Stage 1 Recommendation")
                if report['readiness_score'] >= 80:
                    st.success(f"**{report['recommendation']}**")
                elif report['readiness_score'] >= 60:
                    st.warning(f"**{report['recommendation']}**")
                else:
                    st.error(f"**{report['recommendation']}**")
                
                # Detailed results
                st.header("3. Detailed Analysis Results")
                
                # Status distribution
                st.subheader("Compliance Status Distribution")
                status_dist = results["status"].value_counts()
                st.bar_chart(status_dist)
                
                # Critical vs Non-critical analysis
                st.subheader("Critical Requirements Analysis")
                critical_summary = results.groupby('is_critical').agg({
                    'status': ['count', 
                              lambda x: (x == 'Compliant').sum(),
                              lambda x: (x == 'Partial').sum(),
                              lambda x: (x == 'Not Found').sum()]
                }).round(0)
                critical_summary.columns = ['Total', 'Compliant', 'Partial', 'Not Found']
                critical_summary.index = ['Non-Critical', 'Critical']
                st.dataframe(critical_summary, use_container_width=True)
                
                # Display full results table
                st.subheader("Detailed Requirement Analysis")
                display_cols = ["clause", "requirement", "status", "evidence", "evidence_document", "is_critical"]
                
                # Add color coding for status
                def color_status(val):
                    if val == "Compliant":
                        return "color: green; font-weight: bold;"
                    elif val == "Partial":
                        return "color: orange; font-weight: bold;"
                    elif val == "Not Found":
                        return "color: red; font-weight: bold;"
                    return ""
                
                # Add color coding for critical items
                def color_critical(val):
                    if val == True:
                        return "background-color: #fff3cd; font-weight: bold;"
                    return ""
                
                styled_df = results[display_cols].style.map(color_status, subset=['status']).map(color_critical, subset=['is_critical'])
                st.dataframe(styled_df, use_container_width=True, height=600)
                
                # Filter options
                st.subheader("Filter Results")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    filter_status = st.multiselect(
                        "Filter by Status:",
                        options=["Compliant", "Partial", "Not Found"],
                        default=["Compliant", "Partial", "Not Found"],
                        key="filter_status"
                    )
                
                with col2:
                    filter_critical = st.selectbox(
                        "Filter by Criticality:",
                        options=["All", "Critical Only", "Non-Critical Only"],
                        key="filter_critical"
                    )
                
                with col3:
                    # Get available document names for filtering
                    available_documents = sorted(results['evidence_document'].unique().tolist())
                    if "" in available_documents:
                        available_documents.remove("")
                        available_documents.insert(0, "(No Evidence Document)")
                    
                    filter_document = st.multiselect(
                        "Filter by Document:",
                        options=available_documents,
                        default=available_documents,
                        key="filter_document"
                    )
                    filter_document_for_df = ["" if name == "(No Evidence Document)" else name for name in filter_document]
                
                # Apply filters
                filtered_df = results.copy()
                
                if filter_status:
                    filtered_df = filtered_df[filtered_df["status"].isin(filter_status)]
                
                if filter_critical == "Critical Only":
                    filtered_df = filtered_df[filtered_df["is_critical"] == True]
                elif filter_critical == "Non-Critical Only":
                    filtered_df = filtered_df[filtered_df["is_critical"] == False]
                
                if filter_document_for_df:
                    filtered_df = filtered_df[filtered_df["evidence_document"].isin(filter_document_for_df)]
                
                if not filtered_df.empty:
                    st.write(f"Showing {len(filtered_df)} of {len(results)} Stage 1 requirements")
                    st.dataframe(filtered_df[display_cols].style.map(color_status, subset=['status']).map(color_critical, subset=['is_critical']), 
                                use_container_width=True, height=400)
                else:
                    st.info("No results match the selected filters")
                
                # Download options
                st.header("4. Export Results")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Full results download
                    csv_full = results.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📥 Download Full Analysis Results",
                        csv_full,
                        "stage1_full_analysis_results.csv",
                        "text/csv"
                    )
                
                with col2:
                    # Filtered results download
                    if not filtered_df.empty:
                        csv_filtered = filtered_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "📥 Download Filtered Results",
                            csv_filtered,
                            "stage1_filtered_results.csv",
                            "text/csv"
                        )
                
                # Executive summary download
                executive_summary = f"""
Stage 1 Readiness Assessment Report
===================================

Overall Readiness Score: {report['readiness_score']:.1f}%
Recommendation: {report['recommendation']}

Summary Metrics:
- Total Requirements: {report['total_requirements']}
- Compliant: {report['compliant_count']} ({report['compliant_count']/report['total_requirements']*100:.1f}%)
- Partial: {report['partial_count']} ({report['partial_count']/report['total_requirements']*100:.1f}%)
- Not Found: {report['not_found_count']} ({report['not_found_count']/report['total_requirements']*100:.1f}%)

Critical Requirements:
- Total Critical: {report['total_critical']}
- Critical Compliant: {report['critical_compliant']} ({report['critical_compliant']/report['total_critical']*100:.1f}%)

Analysis completed on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
                """
                
                st.download_button(
                    "📄 Download Executive Summary",
                    executive_summary,
                    "stage1_executive_summary.txt",
                    "text/plain"
                )

    elif not uploaded_files:
        st.info("👆 Please upload your organization's documentation to begin the Stage 1 analysis.")

if __name__ == "__main__":
    stage1_main()
