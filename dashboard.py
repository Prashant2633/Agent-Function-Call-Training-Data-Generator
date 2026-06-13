"""
dashboard.py — Streamlit visual dashboard for the training data pipeline.

Launch with: streamlit run dashboard.py
"""

import json
import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

load_dotenv(override=True)

st.set_page_config(
    page_title="Agent Training Data Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for dark theme enhancements
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%);
        border: 1px solid #3d3d5c;
        border-radius: 12px;
        padding: 20px;
        margin: 8px 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e1e2e;
        border-radius: 8px;
        padding: 8px 20px;
    }
    .highlight { color: #7c85f5; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_db_connection():
    """Cached DB connection."""
    try:
        from sqlalchemy import create_engine, text
        db_url = os.getenv("DATABASE_URL", "postgresql://agent:agentpass@localhost:54321/agent_training")
        engine = create_engine(db_url, pool_pre_ping=True)
        return engine, None
    except Exception as e:
        return None, str(e)


def run_query(sql: str, params: dict = None):
    """Run SQL and return list of dicts."""
    engine, err = get_db_connection()
    if err or engine is None:
        return [], err
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            result = conn.execute(text(sql), params or {})
            cols = result.keys()
            return [dict(zip(cols, row)) for row in result.fetchall()], None
    except Exception as e:
        return [], str(e)


def overview_tab():
    st.header("📊 Pipeline Overview")

    rows, err = run_query("""
        SELECT
            COUNT(*) AS total_examples,
            COUNT(*) FILTER (WHERE is_valid = true) AS valid_examples,
            COUNT(*) FILTER (WHERE is_valid = false) AS invalid_examples
        FROM examples
    """)
    score_rows, _ = run_query("""
        SELECT
            COUNT(*) FILTER (WHERE quality_tier = 'high') AS high_quality,
            COUNT(*) FILTER (WHERE quality_tier = 'medium') AS medium_quality,
            COUNT(*) FILTER (WHERE quality_tier = 'low') AS low_quality,
            AVG(composite_score) AS avg_score
        FROM scores
    """)
    pair_rows, _ = run_query("SELECT COUNT(*) AS total_pairs FROM preference_pairs")
    instr_rows, _ = run_query("SELECT COUNT(*) AS total_instructions FROM instructions")

    if err:
        st.error(f"🔴 Database not connected: {err}")
        st.info("Start PostgreSQL with: `docker-compose up -d db` and run the pipeline first.")
        return

    ex = rows[0] if rows else {}
    sc = score_rows[0] if score_rows else {}
    pp = pair_rows[0] if pair_rows else {}
    ins = instr_rows[0] if instr_rows else {}

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("📝 Instructions", f"{ins.get('total_instructions', 0):,}")
    col2.metric("⚙️ Examples Generated", f"{ex.get('total_examples', 0):,}")
    col3.metric("✅ Valid Examples", f"{ex.get('valid_examples', 0):,}")
    col4.metric("⭐ High Quality", f"{sc.get('high_quality', 0):,}")
    col5.metric("🤝 Preference Pairs", f"{pp.get('total_pairs', 0):,}")
    avg = sc.get('avg_score')
    col6.metric("🎯 Avg Score", f"{avg:.4f}" if avg else "N/A")

    st.divider()
    st.subheader("📈 Generation Progress by Domain")
    domain_rows, _ = run_query("""
        SELECT domain,
               COUNT(*) AS total,
               COUNT(*) FILTER (WHERE is_valid=true) AS valid
        FROM examples GROUP BY domain ORDER BY total DESC
    """)
    if domain_rows:
        import pandas as pd
        df = pd.DataFrame(domain_rows)
        st.bar_chart(df.set_index("domain")[["total", "valid"]])


def quality_tab():
    st.header("⭐ Quality Distribution")

    rows, err = run_query("""
        SELECT e.domain, s.quality_tier, s.composite_score,
               s.schema_correctness, s.argument_completeness,
               s.intent_alignment, s.hallucination_score, s.chain_coherence
        FROM examples e JOIN scores s ON s.example_id = e.id
        WHERE e.is_valid = true
        ORDER BY s.composite_score DESC
    """)

    if err:
        st.error(f"DB error: {err}")
        return
    if not rows:
        st.info("No scored examples yet. Run the pipeline first.")
        return

    import pandas as pd
    df = pd.DataFrame(rows)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Quality Tier Counts")
        tier_counts = df["quality_tier"].value_counts().reset_index()
        tier_counts.columns = ["Tier", "Count"]
        st.bar_chart(tier_counts.set_index("Tier"))

    with col2:
        st.subheader("Score Distribution by Domain")
        domain_scores = df.groupby("domain")["composite_score"].mean().reset_index()
        domain_scores.columns = ["Domain", "Avg Score"]
        st.bar_chart(domain_scores.set_index("Domain"))

    st.divider()
    st.subheader("🔍 Score Axis Breakdown")
    axes = ["schema_correctness", "argument_completeness", "intent_alignment", "hallucination_score", "chain_coherence"]
    axis_means = df[axes].mean().reset_index()
    axis_means.columns = ["Axis", "Mean Score"]
    axis_means["Axis"] = axis_means["Axis"].str.replace("_", " ").str.title()
    st.bar_chart(axis_means.set_index("Axis"))


def examples_tab():
    st.header("📄 Examples Browser")

    col1, col2, col3 = st.columns(3)
    with col1:
        domain_filter = st.selectbox("Domain", [
            "all", "calendar", "search", "code_exec", "crm", "weather",
            "finance", "email", "files", "notifications", "maps", "tasks", "database"
        ])
    with col2:
        type_filter = st.selectbox("Example Type", ["all", "1 (Single)", "2 (Chain)", "3 (Ambiguous)", "4 (Parallel)"])
    with col3:
        quality_filter = st.selectbox("Quality Tier", ["all", "high", "medium", "low"])

    where_clauses = ["e.is_valid = true"]
    if domain_filter != "all":
        where_clauses.append(f"e.domain = '{domain_filter}'")
    if type_filter != "all":
        type_num = type_filter.split(" ")[0]
        where_clauses.append(f"e.example_type = {type_num}")
    if quality_filter != "all":
        where_clauses.append(f"s.quality_tier = '{quality_filter}'")

    where_sql = " AND ".join(where_clauses)

    rows, err = run_query(f"""
        SELECT e.id, i.text AS instruction_text, e.domain, e.example_type,
               e.difficulty, e.generator, e.tool_calls_json AS tool_calls, e.conversation_json AS conversation,
               s.composite_score, s.quality_tier
        FROM examples e 
        JOIN scores s ON s.example_id = e.id
        JOIN instructions i ON i.id = e.instruction_id
        WHERE {where_sql}
        ORDER BY s.composite_score DESC
        LIMIT 200
    """)

    if err:
        st.error(f"DB error: {err}")
        return
    if not rows:
        st.info("No examples match the current filters.")
        return

    st.write(f"Showing {len(rows)} examples")

    for i, row in enumerate(rows[:20]):
        score = row.get('composite_score', 0)
        tier = row.get('quality_tier', '')
        tier_emoji = {"high": "⭐", "medium": "🟡", "low": "🟠"}.get(tier, "")
        with st.expander(f"{tier_emoji} [{row['domain']}] {row['instruction_text'][:80]}... | Score: {score:.3f}"):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**Instruction:** {row['instruction_text']}")
                st.write(f"**Domain:** {row['domain']} | **Type:** {row['example_type']} | **Generator:** {row['generator']}")

                st.write("**Conversation:**")
                conv = json.loads(row['conversation']) if isinstance(row['conversation'], str) else row['conversation']
                for turn in (conv or []):
                    role = turn.get('role', '')
                    content = turn.get('content', '')
                    tool_calls = turn.get('tool_calls', [])
                    if role == 'user':
                        st.markdown(f"> 👤 **User:** {content}")
                    elif role == 'assistant' and tool_calls:
                        for tc in tool_calls:
                            st.markdown(f"> 🤖 **Tool Call:** `{tc['name']}`")
                            st.json(tc.get('arguments', {}))
                    elif role == 'assistant':
                        st.markdown(f"> 🤖 **Assistant:** {content}")
                    elif role == 'tool':
                        st.markdown(f"> 🔧 **Tool Result:** {content[:200]}")
            with col2:
                st.metric("Score", f"{score:.3f}")
                st.write(f"**Tier:** {tier_emoji} {tier}")


def domain_analysis_tab():
    st.header("🌐 Domain Analysis")

    rows, err = run_query("""
        SELECT
            e.domain,
            COUNT(*) AS total,
            COUNT(*) FILTER (WHERE e.is_valid=true) AS valid,
            COUNT(*) FILTER (WHERE s.quality_tier='high') AS high_quality,
            COUNT(*) FILTER (WHERE s.quality_tier='medium') AS medium_quality,
            ROUND(AVG(s.composite_score)::numeric, 4) AS avg_score
        FROM examples e
        LEFT JOIN scores s ON s.example_id = e.id
        GROUP BY e.domain
        ORDER BY total DESC
    """)

    if err:
        st.error(f"DB error: {err}")
        return
    if not rows:
        st.info("No data yet.")
        return

    import pandas as pd
    df = pd.DataFrame(rows)
    df.columns = ["Domain", "Total", "Valid", "High Quality", "Medium Quality", "Avg Score"]
    st.dataframe(df, use_container_width=True)

    st.subheader("Examples by Type per Domain")
    type_rows, _ = run_query("""
        SELECT domain, example_type, COUNT(*) AS count
        FROM examples
        WHERE is_valid = true
        GROUP BY domain, example_type
        ORDER BY domain, example_type
    """)
    if type_rows:
        df2 = pd.DataFrame(type_rows)
        df_pivot = df2.pivot(index="domain", columns="example_type", values="count").fillna(0)
        df_pivot.columns = [f"Type {c}" for c in df_pivot.columns]
        st.dataframe(df_pivot, use_container_width=True)


def failure_analysis_tab():
    st.header("❌ Failure Analysis")

    rows, err = run_query("""
        SELECT failure_mode, generator, COUNT(*) AS count
        FROM failures
        GROUP BY failure_mode, generator
        ORDER BY count DESC
    """)

    if err:
        st.error(f"DB error: {err}")
        return
    if not rows:
        st.info("No failures recorded yet.")
        return

    import pandas as pd
    df = pd.DataFrame(rows)
    df.columns = ["Failure Mode", "Generator", "Count"]
    st.dataframe(df, use_container_width=True)

    st.subheader("Failure Counts by Mode")
    mode_counts = df.groupby("Failure Mode")["Count"].sum().reset_index()
    mode_counts = mode_counts.sort_values("Count", ascending=False)
    st.bar_chart(mode_counts.set_index("Failure Mode"))


def main():
    st.title("🤖 Agent Function-Call Training Data Dashboard")
    st.caption("Real-time visualization of the Groq + Gemini data generation pipeline")

    # Sidebar for API keys (useful for deployed configurations)
    st.sidebar.header("🔑 API Credentials")
    st.sidebar.write("Pre-loaded from `.env` locally. Paste keys here when deployed.")

    default_groq = os.getenv("GROQ_API_KEY", "")
    if default_groq == "your_groq_api_key_here":
        default_groq = ""

    default_gemini = os.getenv("GOOGLE_API_KEY", "")
    if default_gemini == "your_google_api_key_here":
        default_gemini = ""

    groq_api = st.sidebar.text_input(
        "Groq API Key (Generator A)",
        value=default_groq,
        type="password",
        help="Register and obtain from console.groq.com"
    )

    gemini_api = st.sidebar.text_input(
        "Gemini API Key (Generator B & Judge)",
        value=default_gemini,
        type="password",
        help="Register and obtain from aistudio.google.com"
    )

    # Set keys globally in memory for the session
    if groq_api.strip():
        os.environ["GROQ_API_KEY"] = groq_api
    if gemini_api.strip():
        os.environ["GOOGLE_API_KEY"] = gemini_api
        import google.generativeai as genai
        genai.configure(api_key=gemini_api)

    st.sidebar.divider()
    st.sidebar.subheader("📡 API Verification")
    
    if groq_api.strip():
        if groq_api.startswith("gsk_"):
            st.sidebar.success("🟢 Groq API: Key Active")
        else:
            st.sidebar.warning("🟡 Groq API: Unexpected format")
    else:
        st.sidebar.error("🔴 Groq API: Key Missing")

    if gemini_api.strip():
        st.sidebar.success("🟢 Gemini API: Key Active")
    else:
        st.sidebar.error("🔴 Gemini API: Key Missing")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "⭐ Quality",
        "📄 Examples",
        "🌐 Domains",
        "❌ Failures",
    ])

    with tab1:
        overview_tab()
    with tab2:
        quality_tab()
    with tab3:
        examples_tab()
    with tab4:
        domain_analysis_tab()
    with tab5:
        failure_analysis_tab()


if __name__ == "__main__":
    main()
