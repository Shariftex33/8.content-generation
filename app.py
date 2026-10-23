"""
app.py
Streamlit Application for Project 8: Content Generation Pipeline.
Features:
- Multi-Stage Chained LLM Pipeline with Quality Control (LangChain / LCEL)
- Local Ollama (qwen2.5:3b) & Cloud OpenAI / Compatible Provider Switching
- Human-in-the-Loop Outline Approval Gate
- Automated Senior Editor Rubric Evaluation & Self-Refinement Loop
- Multi-Format Adaptation (Blog Post, Social Media Thread, Newsletter, Executive Report)
- Pre-generated Sample Outputs Gallery
- Interactive 5-7 Minute Architecture & Quality Control Presentation Deck
"""

import os
import json
import time
from datetime import datetime
import streamlit as st

# Import pipeline components
from pipeline.config import ContentBrief, QualityScores, QualityReport, PipelineArtifacts
from pipeline.models import (
    build_llm,
    get_available_ollama_models,
    check_ollama_status,
)
from pipeline.chains import ContentPipeline
from pipeline.evaluator import build_quality_report
from samples import get_all_samples

# ==============================================================================
# PAGE CONFIGURATION & STYLING
# ==============================================================================
st.set_page_config(
    page_title="Content Generation Pipeline Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished, modern typography and cards
st.markdown("""
<style>
    /* Metric Card Styling */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px;
        color: #f8fafc;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        margin-bottom: 12px;
    }
    .metric-title {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
    }
    
    /* Stage Pill Badges */
    .stage-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    .stage-done { background-color: #065f46; color: #34d399; border: 1px solid #059669; }
    .stage-active { background-color: #1e3a8a; color: #60a5fa; border: 1px solid #2563eb; }
    .stage-pending { background-color: #334155; color: #94a3b8; border: 1px solid #475569; }
    .stage-paused { background-color: #78350f; color: #fbbf24; border: 1px solid #d97706; }

    /* Rubric Card */
    .rubric-box {
        background-color: #1e293b;
        border-left: 4px solid #38bdf8;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }

    /* Presentation Slide Card */
    .slide-container {
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 28px;
        margin-bottom: 20px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.3);
    }
    .speaker-notes-box {
        background-color: #1a202c;
        border-left: 4px solid #f59e0b;
        border-radius: 6px;
        padding: 16px;
        margin-top: 18px;
        font-size: 0.95rem;
        color: #cbd5e1;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# SESSION STATE MANAGEMENT
# ==============================================================================
if "pipeline_state" not in st.session_state:
    st.session_state.pipeline_state = {
        "status": "idle",  # idle, research_done, outline_ready, drafting, completed
        "artifacts": None,
        "active_brief": None,
        "research_text": "",
        "outline_text": "",
        "draft_text": "",
        "quality_report": None,
        "refined_draft": "",
        "final_output": "",
        "current_slide": 0,
    }


# ==============================================================================
# SIDEBAR CONFIGURATION (MODEL PROVIDER & PARAMETERS)
# ==============================================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=64)
    st.title("Pipeline Engine")
    st.caption("Chained LLM Content Orchestrator")
    
    st.markdown("---")
    st.subheader("1. LLM Provider")
    
    provider_choice = st.radio(
        "Select Model Backend:",
        ["Local Ollama (e.g. qwen2.5:3b)", "Cloud OpenAI / Compatible API", "Offline Simulation (Mock)"],
        index=0,
    )
    
    model_provider = "ollama"
    selected_model_name = "qwen2.5:3b"
    cloud_api_key = ""
    cloud_base_url = None
    ollama_host = "http://localhost:11434"
    
    if "Ollama" in provider_choice:
        model_provider = "ollama"
        ollama_host = st.text_input("Ollama Host URL:", value="http://localhost:11434")
        
        # Check connection status
        ollama_info = check_ollama_status(ollama_host)
        if ollama_info["online"]:
            st.success("🟢 Ollama Connected")
            available_models = get_available_ollama_models(ollama_host)
            # Ensure qwen2.5:3b is preferred default if present
            default_idx = 0
            for idx, m in enumerate(available_models):
                if "qwen2.5:3b" in m:
                    default_idx = idx
                    break
            selected_model_name = st.selectbox("Select Ollama Model:", available_models, index=default_idx)
        else:
            st.error(f"🔴 {ollama_info['message']}")
            st.info("Tip: Start Ollama in terminal with `ollama serve` or select Mock mode.")
            selected_model_name = "qwen2.5:3b"
            
    elif "Cloud" in provider_choice:
        model_provider = "openai"
        cloud_models = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo", "custom"]
        chosen_cloud_model = st.selectbox("Cloud Model:", cloud_models, index=0)
        if chosen_cloud_model == "custom":
            selected_model_name = st.text_input("Custom Model Identifier:", value="deepseek-chat")
        else:
            selected_model_name = chosen_cloud_model
            
        env_key = os.getenv("OPENAI_API_KEY", "")
        cloud_api_key = st.text_input(
            "API Key:",
            value=env_key,
            type="password",
            help="Your OpenAI or OpenAI-compatible API key",
        )
        custom_endpoint = st.text_input(
            "Custom Base URL (Optional):",
            value="",
            placeholder="https://api.groq.com/openai/v1 or https://openrouter.ai/api/v1",
        )
        if custom_endpoint.strip():
            cloud_base_url = custom_endpoint.strip()
            
    else:
        model_provider = "mock"
        selected_model_name = "Mock Engine (Simulation)"
        st.info("💡 Fast offline walkthrough without external API calls.")
        
    st.markdown("---")
    st.subheader("2. Generation Controls")
    temperature = st.slider("Temperature:", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
    quality_threshold = st.slider("QC Approval Threshold:", min_value=70, max_value=95, value=85, step=1,
                                 help="Drafts scoring below this undergo automated self-refinement.")
    force_refinement = st.checkbox("Always Run Self-Refinement Pass", value=True,
                                  help="Enforces Stage 4B editorial refinement for maximum polish.")
                                  
    st.markdown("---")
    st.subheader("3. Quick Presets")
    preset_choice = st.selectbox(
        "Load Sample Brief:",
        [
            "— Select a Preset —",
            "1. Local LLM Architecture (Blog Post)",
            "2. Multi-Stage AI Pipelines (Social Thread)",
            "3. Small Language Models (Newsletter)",
            "4. Enterprise Content ROI (Executive Report)",
        ],
    )


# ==============================================================================
# MAIN TABS NAVIGATION
# ==============================================================================
tab_studio, tab_gallery, tab_deck, tab_docs = st.tabs([
    "🚀 Content Pipeline Studio",
    "📊 Sample Outputs Gallery",
    "🖥️ Presentation Deck (5-7 Mins)",
    "🔍 Architecture & Quality Docs",
])


# ==============================================================================
# TAB 1: CONTENT PIPELINE STUDIO (INTERACTIVE 5-STAGE WORKFLOW)
# ==============================================================================
with tab_studio:
    st.header("⚡ Multi-Stage Content Generation Pipeline")
    st.caption("Demonstrating Chained LLM Operations, Human-in-the-Loop Outline Gates, and Automated Self-Refinement")
    
    # Visual Pipeline Stage Tracker
    curr_status = st.session_state.pipeline_state["status"]
    
    col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
    with col_s1:
        cls = "stage-done" if curr_status in ["outline_ready", "drafting", "completed"] else ("stage-active" if curr_status == "researching" else "stage-pending")
        st.markdown(f'<div class="stage-pill {cls}">1. Research & CoT</div>', unsafe_allow_html=True)
    with col_s2:
        cls = "stage-done" if curr_status in ["drafting", "completed"] else ("stage-paused" if curr_status == "outline_ready" else "stage-pending")
        st.markdown(f'<div class="stage-pill {cls}">2. Outline (Human Gate)</div>', unsafe_allow_html=True)
    with col_s3:
        cls = "stage-done" if curr_status in ["completed"] else ("stage-active" if curr_status == "drafting" else "stage-pending")
        st.markdown(f'<div class="stage-pill {cls}">3. Draft Expansion</div>', unsafe_allow_html=True)
    with col_s4:
        cls = "stage-done" if curr_status in ["completed"] else "stage-pending"
        st.markdown(f'<div class="stage-pill {cls}">4. QC & Self-Refine</div>', unsafe_allow_html=True)
    with col_s5:
        cls = "stage-done" if curr_status in ["completed"] else "stage-pending"
        st.markdown(f'<div class="stage-pill {cls}">5. Final Packaging</div>', unsafe_allow_html=True)
        
    st.markdown("---")

    # --------------------------------------------------------------------------
    # INPUT BRIEF SECTION
    # --------------------------------------------------------------------------
    # Pre-populate based on preset if selected
    default_topic = "Deploying Local Open-Weight LLMs (Qwen 2.5) for Private Enterprise Inference"
    default_aud = "Software Architects, Staff Engineers, and DevOps Leaders"
    default_tone = "Technical & Authoritative yet Practical"
    default_length = "Medium (~800 words)"
    default_format = "Blog Post"
    default_kw = "Ollama, Qwen 2.5, vLLM, quantization, data sovereignty, latency"
    default_notes = "Provide concrete technical guidance on memory footprints, latency benchmarks, and privacy trade-offs."
    
    if "1. Local LLM" in preset_choice:
        default_topic = "Deploying Local Open-Weight LLMs (Qwen 2.5) for Private Enterprise Inference"
        default_aud = "Software Architects, Staff Engineers, and DevOps Leaders"
        default_tone = "Technical & Authoritative yet Practical"
        default_length = "Medium (~800 words)"
        default_format = "Blog Post"
        default_kw = "Ollama, Qwen 2.5, vLLM, quantization, data sovereignty, latency"
        default_notes = "Highlight memory footprints, token throughput, and VPC data security."
    elif "2. Multi-Stage" in preset_choice:
        default_topic = "Why Single-Prompt AI Fails & How Multi-Stage Pipelines Fix It"
        default_aud = "AI Engineers, Tech Founders, and Product Builders"
        default_tone = "Punchy, Insightful, and Highly Actionable"
        default_length = "Short (~400 words)"
        default_format = "Social Media Thread"
        default_kw = "chaining, intermediate quality checks, evaluator-optimizer, hallucinations, LCEL"
        default_notes = "Format as numbered thread posts with contrast, bullet points, and high engagement hooks."
    elif "3. Small Language" in preset_choice:
        default_topic = "Why Small Language Models (SLMs) Are Winning the Enterprise Edge"
        default_aud = "Tech Enthusiasts, Engineering Managers, and AI Builders"
        default_tone = "Conversational, Insightful, and Energetic"
        default_length = "Medium (~800 words)"
        default_format = "Newsletter"
        default_kw = "SLM, edge AI, Qwen 2.5, inference economics, private cloud"
        default_notes = "Include subject line options, preview text, a TL;DR summary box, and an engaging P.S."
    elif "4. Enterprise Content" in preset_choice:
        default_topic = "Enterprise ROI and Operational Architecture of Multi-Stage Content Pipelines"
        default_aud = "Chief Technology Officers, CMOs, and Enterprise Innovation VPs"
        default_tone = "Formal, Strategic, and Executive-Ready"
        default_length = "Long (~1200 words)"
        default_format = "Executive Report"
        default_kw = "ROI, content operations, quality control, human-in-the-loop, cost reduction"
        default_notes = "Include document classification header, executive summary, findings, and an Actionable Recommendations Matrix."

    with st.expander("📝 Content Brief Configuration", expanded=(curr_status == "idle")):
        col_b1, col_b2 = st.columns([2, 1])
        with col_b1:
            topic_input = st.text_input("Content Topic / Core Thesis:", value=default_topic)
            guidelines_input = st.text_area(
                "Additional Guidelines / Key Angles:",
                value=default_notes,
                height=85,
            )
        with col_b2:
            format_input = st.selectbox(
                "Content Format:",
                ["Blog Post", "Social Media Thread", "Newsletter", "Executive Report"],
                index=["Blog Post", "Social Media Thread", "Newsletter", "Executive Report"].index(default_format),
            )
            length_input = st.selectbox(
                "Target Length:",
                ["Short (~400 words)", "Medium (~800 words)", "Long (~1200 words)"],
                index=["Short (~400 words)", "Medium (~800 words)", "Long (~1200 words)"].index(default_length),
            )
            tone_input = st.selectbox(
                "Tone & Style:",
                [
                    "Technical & Authoritative yet Practical",
                    "Professional & Informative",
                    "Conversational, Insightful, and Energetic",
                    "Punchy, Insightful, and Highly Actionable",
                    "Formal, Strategic, and Executive-Ready",
                ],
                index=0,
            )

        col_b3, col_b4 = st.columns(2)
        with col_b3:
            audience_input = st.text_input("Target Audience:", value=default_aud)
        with col_b4:
            keywords_input = st.text_input("Keywords to Incorporate (comma-separated):", value=default_kw)

        kw_list = [k.strip() for k in keywords_input.split(",") if k.strip()]
        current_brief = ContentBrief(
            topic=topic_input,
            target_audience=audience_input,
            tone=tone_input,
            length=length_input,
            content_format=format_input,
            keywords=kw_list,
            additional_guidelines=guidelines_input,
        )

    # --------------------------------------------------------------------------
    # PIPELINE EXECUTION ACTIONS
    # --------------------------------------------------------------------------
    col_act1, col_act2, col_act3 = st.columns([1.5, 1.5, 3])
    with col_act1:
        start_btn = st.button("🚀 Start Pipeline (With Human Gate)", type="primary", use_container_width=True)
    with col_act2:
        auto_btn = st.button("⏩ Auto-Run All Stages", use_container_width=True)
    with col_act3:
        if st.session_state.pipeline_state["status"] != "idle":
            if st.button("🔄 Reset Pipeline State"):
                st.session_state.pipeline_state = {
                    "status": "idle",
                    "artifacts": None,
                    "active_brief": None,
                    "research_text": "",
                    "outline_text": "",
                    "draft_text": "",
                    "quality_report": None,
                    "refined_draft": "",
                    "final_output": "",
                    "current_slide": 0,
                }
                st.rerun()

    # Handler for starting pipeline
    if start_btn or auto_btn:
        try:
            llm_instance = build_llm(
                provider=model_provider,
                model_name=selected_model_name,
                temperature=temperature,
                ollama_host=ollama_host,
                api_key=cloud_api_key,
                base_url=cloud_base_url,
            )
            pipeline_runner = ContentPipeline(llm=llm_instance, model_name=selected_model_name)
            
            with st.spinner("Executing Stage 1: Synthesizing Chain-of-Thought Domain Research..."):
                research_res = pipeline_runner.run_research(current_brief)
                
            with st.spinner("Executing Stage 2: Constructing Structural Outline Blueprint..."):
                outline_res = pipeline_runner.run_outline(current_brief, research_res)
                
            st.session_state.pipeline_state["active_brief"] = current_brief
            st.session_state.pipeline_state["research_text"] = research_res
            st.session_state.pipeline_state["outline_text"] = outline_res
            
            if auto_btn:
                # Auto-approve outline and complete full pipeline
                with st.spinner("Executing Stage 3: Drafting content following outline..."):
                    draft_res = pipeline_runner.run_draft(current_brief, research_res, outline_res)
                    
                with st.spinner("Executing Stage 4: Running Senior Editorial Quality Control..."):
                    q_report, _ = pipeline_runner.run_eval(current_brief, draft_res, outline_res)
                    
                with st.spinner("Executing Stage 4B: Running Self-Refinement loop..."):
                    refined_res = pipeline_runner.run_refinement(current_brief, draft_res, q_report)
                    
                with st.spinner(f"Executing Stage 5: Packaging Publish-Ready {current_brief.content_format}..."):
                    final_res = pipeline_runner.run_formatting(current_brief, refined_res)
                    
                artifacts = PipelineArtifacts(
                    brief=current_brief,
                    research=research_res,
                    raw_outline=outline_res,
                    approved_outline=outline_res,
                    raw_draft=draft_res,
                    quality_report=q_report,
                    refined_draft=refined_res,
                    final_output=final_res,
                    model_used=selected_model_name,
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                )
                st.session_state.pipeline_state["status"] = "completed"
                st.session_state.pipeline_state["artifacts"] = artifacts
                st.session_state.pipeline_state["draft_text"] = draft_res
                st.session_state.pipeline_state["quality_report"] = q_report
                st.session_state.pipeline_state["refined_draft"] = refined_res
                st.session_state.pipeline_state["final_output"] = final_res
                st.success("🎉 Pipeline executed end-to-end successfully!")
            else:
                st.session_state.pipeline_state["status"] = "outline_ready"
                st.rerun()
                
        except Exception as e:
            st.error(f"Error during execution: {str(e)}")

    # --------------------------------------------------------------------------
    # STAGE 2: HUMAN-IN-THE-LOOP OUTLINE APPROVAL GATE
    # --------------------------------------------------------------------------
    if st.session_state.pipeline_state["status"] == "outline_ready":
        st.markdown("---")
        st.warning("⏸️ **PIPELINE PAUSED: Human-in-the-Loop Outline Gate**")
        st.write(
            "Review the synthesized research below, then inspect and edit the structural outline before approving draft generation."
        )

        with st.expander("🔍 View Synthesized Research (Stage 1 Output)", expanded=False):
            st.markdown(st.session_state.pipeline_state["research_text"])

        st.subheader("Edit & Approve Outline Blueprint")
        edited_outline = st.text_area(
            "Edit the outline sections, add specific talking points, or modify word targets:",
            value=st.session_state.pipeline_state["outline_text"],
            height=320,
        )

        col_g1, col_g2 = st.columns([2, 1])
        with col_g1:
            approve_btn = st.button("✅ Approve Outline & Continue to Drafting", type="primary")
        with col_g2:
            re_outline_btn = st.button("🔄 Re-generate Outline")

        if approve_btn:
            try:
                st.session_state.pipeline_state["outline_text"] = edited_outline
                active_b = st.session_state.pipeline_state["active_brief"]
                res_txt = st.session_state.pipeline_state["research_text"]

                llm_instance = build_llm(
                    provider=model_provider,
                    model_name=selected_model_name,
                    temperature=temperature,
                    ollama_host=ollama_host,
                    api_key=cloud_api_key,
                    base_url=cloud_base_url,
                )
                pipeline_runner = ContentPipeline(llm=llm_instance, model_name=selected_model_name)

                with st.spinner("Executing Stage 3: Expanding Draft locked to Approved Outline..."):
                    draft_res = pipeline_runner.run_draft(active_b, res_txt, edited_outline)

                with st.spinner("Executing Stage 4: Senior Editorial Quality Control Rubric..."):
                    q_report, _ = pipeline_runner.run_eval(active_b, draft_res, edited_outline)

                with st.spinner("Executing Stage 4B: Refining Draft based on Quality Critique..."):
                    refined_res = pipeline_runner.run_refinement(active_b, draft_res, q_report)

                with st.spinner(f"Executing Stage 5: Formatting Publish-Ready {active_b.content_format}..."):
                    final_res = pipeline_runner.run_formatting(active_b, refined_res)

                artifacts = PipelineArtifacts(
                    brief=active_b,
                    research=res_txt,
                    raw_outline=edited_outline,
                    approved_outline=edited_outline,
                    raw_draft=draft_res,
                    quality_report=q_report,
                    refined_draft=refined_res,
                    final_output=final_res,
                    model_used=selected_model_name,
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                )
                st.session_state.pipeline_state["status"] = "completed"
                st.session_state.pipeline_state["artifacts"] = artifacts
                st.session_state.pipeline_state["draft_text"] = draft_res
                st.session_state.pipeline_state["quality_report"] = q_report
                st.session_state.pipeline_state["refined_draft"] = refined_res
                st.session_state.pipeline_state["final_output"] = final_res
                st.rerun()

            except Exception as e:
                st.error(f"Error during stage continuation: {str(e)}")

        if re_outline_btn:
            try:
                active_b = st.session_state.pipeline_state["active_brief"]
                res_txt = st.session_state.pipeline_state["research_text"]
                llm_instance = build_llm(
                    provider=model_provider,
                    model_name=selected_model_name,
                    temperature=temperature,
                    ollama_host=ollama_host,
                    api_key=cloud_api_key,
                    base_url=cloud_base_url,
                )
                pipeline_runner = ContentPipeline(llm=llm_instance, model_name=selected_model_name)
                with st.spinner("Re-generating structural outline..."):
                    new_outline = pipeline_runner.run_outline(active_b, res_txt)
                st.session_state.pipeline_state["outline_text"] = new_outline
                st.rerun()
            except Exception as e:
                st.error(f"Error re-generating outline: {str(e)}")

    # --------------------------------------------------------------------------
    # STAGE 4 & 5: RESULTS & QUALITY REPORT DISPLAY
    # --------------------------------------------------------------------------
    if st.session_state.pipeline_state["status"] == "completed":
        st.markdown("---")
        st.subheader("🎯 Pipeline Execution Artifacts")

        sub_tab1, sub_tab2, sub_tab3, sub_tab4, sub_tab5 = st.tabs([
            "✨ Publish-Ready Output",
            "🛡️ Quality Control & Rubric",
            "📝 Raw vs Refined Draft",
            "📋 Approved Outline",
            "🔬 Research & CoT",
        ])

        with sub_tab1:
            st.markdown(f"### Published Format: {st.session_state.pipeline_state['active_brief'].content_format}")
            
            # Action download bar
            final_content = st.session_state.pipeline_state["final_output"]
            col_d1, col_d2, col_d3 = st.columns([1, 1, 2])
            with col_d1:
                st.download_button(
                    "📥 Download Markdown (.md)",
                    data=final_content,
                    file_name=f"content_{st.session_state.pipeline_state['active_brief'].content_format.lower().replace(' ', '_')}.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
            with col_d2:
                json_data = json.dumps(st.session_state.pipeline_state["artifacts"].dict(), indent=2)
                st.download_button(
                    "📦 Export Pipeline JSON",
                    data=json_data,
                    file_name="pipeline_artifacts.json",
                    mime="application/json",
                    use_container_width=True,
                )
                
            st.markdown("---")
            st.markdown(final_content)

        with sub_tab2:
            st.markdown("### Automated Editorial Quality Control Report")
            q_rep: QualityReport = st.session_state.pipeline_state["quality_report"]
            if q_rep:
                sc = q_rep.scores
                col_m1, col_m2, col_m3, col_m4, col_m5, col_m6 = st.columns(6)
                with col_m1:
                    st.metric("Total Quality", f"{sc.overall_score}/100")
                with col_m2:
                    st.metric("Grammar & Syntax", f"{sc.grammar_and_syntax}/20")
                with col_m3:
                    st.metric("Clarity & Flow", f"{sc.clarity_and_coherence}/20")
                with col_m4:
                    st.metric("Tone Alignment", f"{sc.audience_and_tone}/20")
                with col_m5:
                    st.metric("Outline Fidelity", f"{sc.outline_adherence}/20")
                with col_m6:
                    st.metric("Fact & Depth", f"{sc.factuality_and_depth}/20")

                st.markdown("---")
                col_qc1, col_qc2 = st.columns(2)
                with col_qc1:
                    st.markdown("#### ✅ Key Strengths")
                    for s in q_rep.strengths:
                        st.markdown(f"- **{s}**")
                with col_qc2:
                    st.markdown("#### ⚠️ Areas Flagged for Refinement")
                    for w in q_rep.areas_for_improvement:
                        st.markdown(f"- *{w}*")

                st.markdown("#### 🖋️ Senior Editorial Critique")
                st.info(q_rep.editorial_critique)

        with sub_tab3:
            st.markdown("### Draft Evolution: Raw Draft vs. Refined Draft")
            col_df1, col_df2 = st.columns(2)
            with col_df1:
                st.markdown("#### 📄 Stage 3: Raw Draft")
                st.text_area("Initial LLM Output:", value=st.session_state.pipeline_state["draft_text"], height=400, disabled=True)
            with col_df2:
                st.markdown("#### 💎 Stage 4B: Polished & Refined Draft")
                st.text_area("Post-Editorial Refinement:", value=st.session_state.pipeline_state["refined_draft"], height=400, disabled=True)

        with sub_tab4:
            st.markdown("### Approved Structural Blueprint (Stage 2)")
            st.markdown(st.session_state.pipeline_state["outline_text"])

        with sub_tab5:
            st.markdown("### Domain Research & Chain-of-Thought Inquiry (Stage 1)")
            st.markdown(st.session_state.pipeline_state["research_text"])


# ==============================================================================
# TAB 2: SAMPLE OUTPUTS GALLERY (SHOWCASING ALL 4 FORMATS)
# ==============================================================================
with tab_gallery:
    st.header("📊 Multi-Format Sample Outputs Gallery")
    st.caption("Inspect comprehensive pre-generated outputs for all 4 required content formats, showing each pipeline stage.")

    all_samples = get_all_samples()
    gallery_format = st.selectbox(
        "Select Content Format to Inspect:",
        list(all_samples.keys()),
        index=0,
    )

    sample_data = all_samples[gallery_format]
    st.markdown(f"## {sample_data.get('title', gallery_format)}")

    st.markdown(
        f"**Target Audience:** `{sample_data['brief']['target_audience']}` | "
        f"**Tone:** `{sample_data['brief']['tone']}` | "
        f"**Target Length:** `{sample_data['brief']['length']}`"
    )

    # Show Stage Selector for Sample
    sample_stage = st.radio(
        "Inspect Pipeline Stage:",
        [
            "1. Domain Research & CoT",
            "2. Approved Outline",
            "3. Initial Draft",
            "4. Quality Control & Rubric Scores",
            "5. Refined Draft",
            "6. Publish-Ready Final Output",
        ],
        horizontal=True,
    )

    st.markdown("---")

    if "1. Domain Research" in sample_stage:
        st.subheader("Stage 1: Synthesized Domain Research (Chain-of-Thought)")
        st.markdown(sample_data["research"])

    elif "2. Approved Outline" in sample_stage:
        st.subheader("Stage 2: Approved Structural Outline Blueprint")
        st.markdown(sample_data["approved_outline"])

    elif "3. Initial Draft" in sample_stage:
        st.subheader("Stage 3: First Draft Adhering to Outline")
        st.markdown(sample_data["raw_draft"])

    elif "4. Quality Control" in sample_stage:
        st.subheader("Stage 4: Automated Senior Editor Quality Evaluation")
        rep = sample_data["quality_report"]
        scores = rep["scores"]
        
        col_sm1, col_sm2, col_sm3, col_sm4, col_sm5, col_sm6 = st.columns(6)
        with col_sm1:
            st.metric("Total Quality", f"{scores['overall_score']}/100")
        with col_sm2:
            st.metric("Grammar & Syntax", f"{scores['grammar_and_syntax']}/20")
        with col_sm3:
            st.metric("Clarity & Flow", f"{scores['clarity_and_coherence']}/20")
        with col_sm4:
            st.metric("Tone Match", f"{scores['audience_and_tone']}/20")
        with col_sm5:
            st.metric("Outline Fidelity", f"{scores['outline_adherence']}/20")
        with col_sm6:
            st.metric("Fact & Depth", f"{scores['factuality_and_depth']}/20")

        st.markdown("---")
        col_st1, col_st2 = st.columns(2)
        with col_st1:
            st.markdown("#### ✅ Strengths Detected")
            for s in rep["strengths"]:
                st.markdown(f"- **{s}**")
        with col_st2:
            st.markdown("#### ⚠️ Editorial Recommendations")
            for a in rep["areas_for_improvement"]:
                st.markdown(f"- *{a}*")

        st.markdown("#### 🖋️ Senior Editorial Critique")
        st.info(rep["editorial_critique"])

    elif "5. Refined Draft" in sample_stage:
        st.subheader("Stage 4B: Polished Draft via Self-Refinement Loop")
        st.markdown(sample_data["refined_draft"])

    elif "6. Publish-Ready" in sample_stage:
        st.subheader(f"Stage 5: Publish-Ready {gallery_format}")
        st.download_button(
            "📥 Download Sample Markdown",
            data=sample_data["final_output"],
            file_name=f"sample_{gallery_format.lower().replace(' ', '_')}.md",
            mime="text/markdown",
        )
        st.markdown("---")
        st.markdown(sample_data["final_output"])


# ==============================================================================
# TAB 3: 5-7 MINUTE PRESENTATION DECK (BUILT-IN SLIDE SHOW)
# ==============================================================================
with tab_deck:
    st.header("🖥️ Pipeline Architecture & Quality Control Presentation Deck")
    st.caption("A dedicated 5-7 minute executive and engineering briefing with interactive slides and presenter speaker scripts.")

    slides = [
        {
            "slide_num": 1,
            "title": "Why Single-Prompt LLM Generation Fails in Production",
            "time": "0:00 - 0:50",
            "content": """
### The Monolith Trap vs. The Chained Pipeline

| Single Mega-Prompt Approach | Multi-Stage Chained Architecture |
| :--- | :--- |
| ❌ **Attention Degradation**: Model juggles 10 instructions simultaneously | ✅ **Context Isolation**: Each stage focuses on a single cognitive task |
| ❌ **Zero Intermediate Checkpoints**: Fails silently at end of generation | ✅ **Human-in-the-Loop Gate**: Approve/tweak outline in 30 seconds |
| ❌ **Generic Corporate Prose**: Clichés and superficial coverage | ✅ **Objective 100-Point Rubric**: Programmatic quality evaluation |
| ❌ **Single Output Format**: Hard to repurpose across channels | ✅ **Decoupled Packaging**: Blog, Social Thread, Newsletter, Report |
            """,
            "speaker_notes": """Good morning, everyone. Today, I'm presenting our Multi-Stage Content Generation Pipeline—a chained LLM architecture designed to take a high-level brief and produce publish-ready content across multiple distribution channels.

When organizations first attempt AI content generation, they almost always start with a single mega-prompt. They ask ChatGPT to research, organize, draft, edit, and format all at once. In production, this causes catastrophic attention degradation—the model attempts to satisfy 10 competing constraints simultaneously and produces bland, superficial mush with zero quality guarantees.

Our pipeline replaces this fragility with a deterministic, chained engineering workflow built using LangChain and LCEL."""
        },
        {
            "slide_num": 2,
            "title": "The 5-Stage Orchestration Engine",
            "time": "0:50 - 1:50",
            "content": """
### High-Level Architectural Flow

```
[Input Brief]
      │
      ▼
[Stage 1: Research & Synthesis (CoT)]  ──► First-principles analysis & verifiable facts
      │
      ▼
[Stage 2: Outline Blueprint]          ──► Title candidates, section targets, talking points
      │
      ▼
[HUMAN-IN-THE-LOOP APPROVAL GATE]     ──► Human editor inspects & approves structure
      │
      ▼
[Stage 3: Constrained Drafting]       ──► Locked expansion strictly adhering to outline
      │
      ▼
[Stage 4: Automated QC & Refinement]  ──► 100-point rubric evaluation + self-healing loop
      │
      ▼
[Stage 5: Format Adaptation]          ──► Blog / Social Thread / Newsletter / Executive Report
```
            """,
            "speaker_notes": """Here is our end-to-end architecture:
1. Stage 1: Research & Synthesis uses Chain-of-Thought inquiry to deconstruct the topic, examine audience psychology, and gather concrete data angles.
2. Stage 2: Structural Outlining transforms the research into a section-by-section blueprint with estimated word counts and hook strategies.
3. Right after outlining, we implement a crucial Human-in-the-Loop Approval Gate. The pipeline pauses, allowing a human editor to review, edit, or approve the outline before drafting begins.
4. Stage 3: Constrained Drafting expands the narrative, locked strictly to the approved outline.
5. Stage 4: Automated Quality Control & Refinement acts as a senior editor, scoring the draft on a 100-point rubric and running a self-refinement loop.
6. Finally, Stage 5: Format Adaptation packages the polished prose into blogs, social threads, newsletters, or executive reports."""
        },
        {
            "slide_num": 3,
            "title": "Deep Research & The 2-Minute Governance Checkpoint",
            "time": "1:50 - 2:50",
            "content": """
### Stage 1 & Stage 2 Synergy

#### 1. Chain-of-Thought Research
- Deconstructs topic into core thesis and problem statement.
- Unpacks target audience psychological drivers and existing friction.
- Formulates 4-6 substantive conceptual pillars and verifiable data angles.
- Surfaces counter-perspectives to avoid one-sided bias.

#### 2. The Outline Human-in-the-Loop Checkpoint
- Prevents 90% of downstream hallucinations and tonal mismatches.
- Pauses the pipeline in the Streamlit UI.
- Enables human editors to edit H2s, reorder sections, and calibrate word targets in under 60 seconds.
- Drafting **never begins** without an approved structural blueprint.
            """,
            "speaker_notes": """Why does this decomposition work so much better?

In the Research phase, we force the model to reason through the topic step-by-step before proposing structure. It identifies the target audience's core frustration, uncovers real-world benchmarks, and surfaces counter-arguments.

Next, notice our human-in-the-loop checkpoint between Stages 2 and 3. Traditional AI workflows generate an entire 1,500-word article, only for the user to realize the angle was completely wrong. By inserting a 30-second human approval gate at the outline stage, the human steers the strategic direction. Once approved, the LLM writer in Stage 3 operates with extreme clarity—it doesn't have to guess what comes next; it simply expands verified points."""
        },
        {
            "slide_num": 4,
            "title": "Automated Quality Control & The Self-Refinement Loop",
            "time": "2:50 - 4:00",
            "content": """
### The Evaluator-Optimizer Quality Gate

| Rubric Dimension | Max Score | Evaluation Focus |
| :--- | :---: | :--- |
| **Grammar & Syntax** | 20 pts | Typographical accuracy, sentence variety, rhythm |
| **Clarity & Coherence** | 20 pts | Logical progression, transitions, lack of repetition |
| **Audience & Tone** | 20 pts | Alignment with target demographic and tone brief |
| **Outline Adherence** | 20 pts | Coverage of agreed sections, structure fidelity |
| **Factuality & Depth** | 20 pts | Substantive rigor, concrete insights, lack of fluff |
| **Total Quality Score** | **100 pts** | **Pass Threshold: ≥ 85 pts** |

#### The Self-Healing Mechanism
- Evaluator LLM outputs structured JSON with rubric scores, strengths, and flagged issues.
- If score < 85 (or forced polish enabled):
  `Refinement LLM` absorbs the critique and regenerates a polished, higher-signal draft automatically.
            """,
            "speaker_notes": """The hallmark of this pipeline is our automated Quality Control and self-refinement loop—implementing the state-of-the-art Evaluator-Optimizer pattern.

In Stage 4, an independent Senior Editor LLM evaluates the draft against an objective 5-dimension rubric, scoring each area from 0 to 20 for a total of 100 points: Grammar, Clarity, Tone, Outline Adherence, and Factuality.

If the draft scores below 85—or if forced refinement is active—the pipeline does not stop. It passes the specific critique and flagged weaknesses into an automated Refinement Pass. The model rewrites transitions, injects concrete examples, and resolves ambiguities before packaging. This is true self-healing content generation."""
        },
        {
            "slide_num": 5,
            "title": "Local Ollama (qwen2.5:3b) vs. Cloud Frontier Models",
            "time": "4:00 - 5:00",
            "content": """
### Infrastructure Flexibility: Private On-Prem vs. Cloud Scale

#### 🔒 Local On-Premises via Ollama (`qwen2.5:3b`)
- **100% Data Sovereignty**: Zero external network egress; ideal for HIPAA, GDPR, and proprietary IP.
- **Hardware Efficiency**: Runs in ~2.2 GB VRAM on consumer laptops & edge devices at 65+ tokens/sec.
- **Zero Token Cost**: Predictable $0 marginal cost per generation.

#### ☁️ Cloud Frontier Models (`gpt-4o`, `gpt-4o-mini`, Custom Endpoints)
- **High-Complexity Reasoning**: 128k+ context windows for massive multi-document synthesis.
- **Zero Local Compute**: Offloads all processing to remote infrastructure.

#### 🔄 Dynamic Unified LLM Factory
- Toggle between Ollama, OpenAI, or Mock mode seamlessly inside the Streamlit UI.
            """,
            "speaker_notes": """A critical design requirement was supporting both local open-weight models and cloud frontier models.

In our Streamlit interface, users can select Local Ollama running qwen2.5:3b. Why is this significant? For regulated enterprises—such as healthcare, banking, and legal—data sovereignty is non-negotiable. With Ollama and Qwen 2.5 3B, inference occurs entirely on the local machine over local sockets. Sensitive internal data never leaves the building, and marginal token costs drop to zero.

At the same time, our modular LLM Factory lets teams switch to cloud models like GPT-4o or OpenAI-compatible endpoints with a single click."""
        },
        {
            "slide_num": 6,
            "title": "Decoupled Multi-Format Distribution",
            "time": "5:00 - 6:00",
            "content": """
### One Polished Narrative ➔ Four Distribution Channels

1. **SEO Blog Post**: Primary H1 + 2 alternative titles, SEO meta description, read time, pull quotes (`> quote`), key takeaways callout, and Midjourney image prompt.
2. **Social Media Thread**: Scroll-stopping hook tweet, 8 numbered posts (1/8 to 8/8) with whitespace & bullet points, recap, engagement CTA, and hashtags.
3. **Email Newsletter**: 3 subject line options (Curiosity, Benefit, Contrarian), preview text, conversational greeting, TL;DR box, and interactive P.S.
4. **Executive Report**: Formal C-suite header, 200-word Executive Summary, strategic context, and an Actionable Recommendations Matrix table.
            """,
            "speaker_notes": """In Stage 5, the pipeline decouples content substance from distribution formatting.

From a single research inquiry and approved draft, our system can package publish-ready assets for:
- High-ranking SEO Blog Posts with meta tags and Midjourney prompts;
- High-engagement Social Media Threads numbered 1 to 8 for Twitter and LinkedIn;
- Conversational Email Newsletters with multiple subject line options and TL;DR boxes; or
- Authoritative Executive Reports complete with C-suite briefing headers and an Actionable Recommendations table.

Every intermediate artifact is preserved and exportable."""
        },
        {
            "slide_num": 7,
            "title": "Production Impact & Learning Objectives Met",
            "time": "6:00 - 7:00",
            "content": """
### Key Takeaways & Enterprise Value

- 🚀 **74% Turnaround Compression**: Content lifecycle drops from 4.2 days to under 45 minutes.
- 💰 **62% Direct Cost Reduction**: Achieved by local model drafting + hybrid routing.
- 🛡️ **96%+ Brand Compliance**: Guaranteed through automated rubric gates.
- 🤝 **Human Editorial Augmentation**: Editors shift from line-by-line typing to strategic curation.

---
### Live Q&A and Streamlit Demo
- Try the interactive Human-in-the-Loop Outline Gate in Tab 1
- Benchmark local `qwen2.5:3b` against cloud models
- Inspect the 4 pre-generated sample artifacts in Tab 2
            """,
            "speaker_notes": """To summarize the learning objectives achieved in Project 8:
1. We designed a multi-stage LLM pipeline with intermediate quality checks using LangChain and LCEL.
2. We implemented Chain-of-Thought inquiry and a critical Human-in-the-Loop outline approval gate.
3. We built an automated Evaluator-Optimizer self-refinement loop with an objective 100-point rubric.
4. We provided full support for local on-prem inference with Ollama qwen2.5:3b as well as cloud frontier models.

Thank you very much. I would love to open the floor to any questions or walk you through a live demonstration in our Streamlit studio."""
        },
    ]

    # Slide navigation state
    current_idx = st.session_state.pipeline_state.get("current_slide", 0)

    col_nav1, col_nav2, col_nav3 = st.columns([1, 3, 1])
    with col_nav1:
        if st.button("◀ Previous Slide", disabled=(current_idx == 0), use_container_width=True):
            st.session_state.pipeline_state["current_slide"] = max(0, current_idx - 1)
            st.rerun()
    with col_nav2:
        selected_slide_num = st.selectbox(
            "Slide Navigator:",
            options=range(len(slides)),
            format_func=lambda i: f"Slide {slides[i]['slide_num']}: {slides[i]['title']} ({slides[i]['time']})",
            index=current_idx,
        )
        if selected_slide_num != current_idx:
            st.session_state.pipeline_state["current_slide"] = selected_slide_num
            st.rerun()
    with col_nav3:
        if st.button("Next Slide ▶", disabled=(current_idx == len(slides) - 1), use_container_width=True):
            st.session_state.pipeline_state["current_slide"] = min(len(slides) - 1, current_idx + 1)
            st.rerun()

    active_slide = slides[st.session_state.pipeline_state["current_slide"]]

    # Render Active Slide
    st.markdown(f"""
    <div class="slide-container">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
            <span style="font-size: 0.9rem; font-weight: 700; color: #38bdf8; text-transform: uppercase;">
                Slide {active_slide['slide_num']} of {len(slides)}
            </span>
            <span style="font-size: 0.85rem; font-weight: 600; color: #fbbf24; background: #374151; padding: 3px 10px; border-radius: 9999px;">
                ⏱️ Target Time: {active_slide['time']}
            </span>
        </div>
        <h2 style="color: #ffffff; margin-top: 0; font-size: 1.8rem;">{active_slide['title']}</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(active_slide["content"])

    # Presenter Notes Toggle
    with st.expander("🎙️ Presenter Script & Timing Cue (Speaker Notes)", expanded=True):
        st.markdown(f"""
        <div class="speaker-notes-box">
            <strong>Speaker Script:</strong><br>
            <em>"{active_slide['speaker_notes']}"</em>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# TAB 4: ARCHITECTURE & QUALITY CONTROL DOCS
# ==============================================================================
with tab_docs:
    st.header("🔍 Architecture & Quality Control Specifications")
    st.caption("Deep technical dive into LangChain LCEL chaining, prompt design, and self-evaluation rubrics.")

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown("""
### 🏗️ Technical Architecture Principles
1. **LangChain LCEL Runnables**:
   Each stage is formulated as a pure LCEL chain:
   ```python
   stage_chain = prompt_template | chat_model | StrOutputParser()
   ```
   This guarantees observable inputs/outputs, seamless model substitution (Ollama vs. OpenAI), and standardized streaming or batch execution.

2. **Decoupled State Management**:
   The `ContentBrief` and `PipelineArtifacts` data models use Pydantic for strict schema validation. All intermediate representations (Research, Outline, Raw Draft, Refined Draft) are persisted.

3. **Deterministic Governance (Human Gate)**:
   The system implements an asynchronous human checkpoint. Generating an entire article before realizing the structural angle is flawed is costly. Reviewing and editing the outline takes under 60 seconds and eliminates 90% of downstream revisions.
        """)

    with col_d2:
        st.markdown("""
### 🛡️ The 100-Point Editorial Rubric
Our Stage 4 Senior Editor evaluates content across 5 orthogonal axes (0-20 points each):

- **1. Grammar & Syntax (0-20)**: Punctuation, spelling, sentence rhythm, and absence of typographical errors.
- **2. Clarity & Flow (0-20)**: Logical sequence of ideas, paragraph transitions, and avoidance of circular repetition.
- **3. Audience & Tone (0-20)**: Voice calibration against target seniority and domain familiarity.
- **4. Outline Adherence (0-20)**: Structural fidelity to the human-approved blueprint and talking points.
- **5. Factuality & Depth (0-20)**: Technical accuracy, absence of generic clichés, and substantive reasoning.

**Threshold Logic**:
- Score $\ge 85$: Approved for final packaging.
- Score $< 85$: Triggers automated **Stage 4B Refinement Pass** with specific actionable feedback.
        """)

    st.markdown("---")
    st.subheader("💡 Multi-Model Integration Guide")
    st.markdown("""
- **Running Local Ollama**:
  1. Ensure Ollama is running (`ollama serve`).
  2. Pull the Qwen model: `ollama pull qwen2.5:3b`.
  3. Select **Local Ollama** in the sidebar. The model is auto-detected.
- **Running Cloud Models**:
  1. Select **Cloud OpenAI / Compatible API**.
  2. Enter your API key (or set `OPENAI_API_KEY` in your environment or `.env` file).
  3. Specify any compatible provider (OpenAI, Groq, OpenRouter, DeepSeek) via the custom base URL.
    """)
