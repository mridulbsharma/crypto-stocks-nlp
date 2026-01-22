"""
CryptoSentinel Interactive Demo UI.

Streamlit application providing an interactive interface for exploring
CryptoSentinel's agentic AI capabilities with full reasoning trace visualization.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
from datetime import datetime

st.set_page_config(
    page_title="CryptoSentinel - Agentic AI Demo",
    page_icon="🔍",
    layout="wide"
)


@st.cache_resource
def load_tools():
    """Load and cache CryptoSentinel tools."""
    from src.context_engine.retriever import PostRetriever
    from src.mcp_server.tools import CryptoSentinelTools
    from src.reasoning.socratic_refine import SocraticRefiner
    
    tools = CryptoSentinelTools()
    tools._ensure_initialized()
    return tools


def main():
    st.title("🔍 CryptoSentinel")
    st.markdown("**Agentic AI for Grounded Crypto & Stock Market Intelligence**")
    
    st.sidebar.header("⚙️ Settings")
    
    mode = st.sidebar.radio(
        "Analysis Mode",
        ["🔎 Search", "📊 Sentiment", "🏷️ Classify", "📝 Generate Brief", "🔬 Full Reasoning"]
    )
    
    min_confidence = st.sidebar.slider(
        "Minimum Confidence",
        0.0, 1.0, 0.5,
        help="Minimum confidence threshold for grounded claims"
    )
    
    require_citations = st.sidebar.checkbox(
        "Require Citations",
        value=True,
        help="Require multiple citations for trend claims"
    )
    
    show_trace = st.sidebar.checkbox(
        "Show Reasoning Trace",
        value=True,
        help="Display the Plan → Execute → Verify → Finalize trace"
    )
    
    try:
        with st.spinner("Loading CryptoSentinel..."):
            tools = load_tools()
        st.sidebar.success("✅ System Ready")
        
        stats = tools._retriever.get_statistics()
        st.sidebar.markdown("---")
        st.sidebar.markdown("**📈 Data Statistics**")
        st.sidebar.write(f"Total Posts: {stats.get('total_posts', 'N/A'):,}")
        if 'posts_by_subreddit' in stats:
            for sub, count in stats['posts_by_subreddit'].items():
                st.sidebar.write(f"- r/{sub}: {count:,}")
                
    except Exception as e:
        st.sidebar.error(f"⚠️ Load Error: {e}")
        st.error("Failed to initialize CryptoSentinel. Check that data files exist.")
        st.stop()
    
    if mode == "🔎 Search":
        render_search_mode(tools)
    elif mode == "📊 Sentiment":
        render_sentiment_mode(tools)
    elif mode == "🏷️ Classify":
        render_classify_mode(tools)
    elif mode == "📝 Generate Brief":
        render_brief_mode(tools, show_trace)
    elif mode == "🔬 Full Reasoning":
        render_reasoning_mode(tools, show_trace)


def render_search_mode(tools):
    """Render the search interface."""
    st.header("🔎 Reddit Post Search")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("Search Query", placeholder="e.g., GME short squeeze")
    with col2:
        subreddit = st.selectbox(
            "Subreddit",
            ["All", "wallstreetbets", "CryptoMoonShots"]
        )
    
    col3, col4 = st.columns(2)
    with col3:
        min_score = st.number_input("Minimum Score", min_value=0, value=0)
    with col4:
        max_results = st.slider("Max Results", 1, 50, 10)
    
    if st.button("🔍 Search", type="primary"):
        if not query:
            st.warning("Please enter a search query")
            return
        
        with st.spinner("Searching..."):
            from src.mcp_server.tools import reddit_search
            
            sub_filter = subreddit if subreddit != "All" else None
            posts = reddit_search(
                query=query,
                subreddit=sub_filter,
                min_score=min_score,
                max_results=max_results,
                tools=tools
            )
        
        if not posts:
            st.info("No results found")
            return
        
        st.success(f"Found {len(posts)} results")
        
        for i, post in enumerate(posts, 1):
            with st.expander(f"**{i}. {post.title[:80]}...** (Score: {post.score})", expanded=i<=3):
                st.markdown(f"**Subreddit:** r/{post.subreddit}")
                st.markdown(f"**Relevance:** {post.relevance:.1%}")
                st.markdown(f"**Date:** {post.datetime or 'Unknown'}")
                if post.text and post.text != 'notexthere':
                    st.markdown("**Content:**")
                    st.markdown(f"> {post.text[:500]}...")
                st.markdown(f"[🔗 View on Reddit]({post.url})")


def render_sentiment_mode(tools):
    """Render the sentiment analysis interface."""
    st.header("📊 Sentiment Aggregation")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        topic = st.text_input("Topic", placeholder="e.g., Bitcoin, Tesla, NFT")
    with col2:
        subreddit = st.selectbox(
            "Subreddit",
            ["All", "wallstreetbets", "CryptoMoonShots"]
        )
    
    if st.button("📊 Analyze Sentiment", type="primary"):
        if not topic:
            st.warning("Please enter a topic")
            return
        
        with st.spinner("Analyzing sentiment..."):
            from src.mcp_server.tools import sentiment_aggregate
            
            sub_filter = subreddit if subreddit != "All" else None
            report = sentiment_aggregate(topic=topic, subreddit=sub_filter, tools=tools)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Posts", report.total_posts)
        with col2:
            st.metric("Positive", report.positive_count, delta=f"{report.positive_count/max(1,report.total_posts)*100:.0f}%")
        with col3:
            st.metric("Negative", report.negative_count)
        with col4:
            label_emoji = {"bullish": "🟢", "bearish": "🔴", "neutral": "⚪"}.get(report.sentiment_label, "⚪")
            st.metric("Sentiment", f"{label_emoji} {report.sentiment_label.upper()}")
        
        st.markdown("### Sentiment Distribution")
        chart_data = pd.DataFrame({
            "Category": ["Positive", "Neutral", "Negative"],
            "Count": [report.positive_count, report.neutral_count, report.negative_count]
        })
        st.bar_chart(chart_data.set_index("Category"))
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🟢 Top Positive Posts")
            for post in report.top_positive:
                st.markdown(f"- [{post.title[:50]}...]({post.url})")
        with col2:
            st.markdown("#### 🔴 Top Negative Posts")
            for post in report.top_negative:
                st.markdown(f"- [{post.title[:50]}...]({post.url})")


def render_classify_mode(tools):
    """Render the classification interface."""
    st.header("🏷️ Post Classification")
    
    title = st.text_input("Post Title", placeholder="Enter a post title to classify")
    text = st.text_area("Post Text (Optional)", placeholder="Enter post content for better accuracy")
    
    if st.button("🏷️ Classify", type="primary"):
        if not title:
            st.warning("Please enter a title")
            return
        
        with st.spinner("Classifying..."):
            from src.mcp_server.tools import classify_post
            result = classify_post(title=title, text=text if text else None, tools=tools)
        
        col1, col2 = st.columns(2)
        with col1:
            class_emoji = "🪙" if result.predicted_class == "crypto" else "📈"
            st.metric("Predicted Class", f"{class_emoji} {result.predicted_class.upper()}")
        with col2:
            st.metric("Confidence", f"{result.confidence:.1%}")
        
        st.markdown("### Probability Distribution")
        prob_data = pd.DataFrame({
            "Class": ["Crypto", "Stocks"],
            "Probability": [result.probabilities["crypto"], result.probabilities["stocks"]]
        })
        st.bar_chart(prob_data.set_index("Class"))
        
        st.markdown("### 🔍 Feature Explanations")
        for exp in result.explanations[:5]:
            direction_emoji = "🪙" if exp.direction == "crypto" else "📈"
            st.markdown(f"- **{exp.feature}** → {direction_emoji} {exp.direction} (weight: {exp.weight:.3f})")


def render_brief_mode(tools, show_trace):
    """Render the brief generation interface."""
    st.header("📝 Generate Grounded Brief")
    
    topic = st.text_input("Topic", placeholder="e.g., What's the sentiment on Bitcoin this week?")
    max_sources = st.slider("Maximum Sources", 3, 10, 5)
    
    if st.button("📝 Generate Brief", type="primary"):
        if not topic:
            st.warning("Please enter a topic")
            return
        
        with st.spinner("Generating grounded brief..."):
            from src.mcp_server.tools import generate_brief
            brief = generate_brief(topic=topic, max_sources=max_sources, tools=tools)
        
        groundedness_color = {"high": "🟢", "medium": "🟡", "low": "🔴", "insufficient": "⛔"}.get(brief.groundedness, "⚪")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Confidence", f"{brief.confidence:.1%}")
        with col2:
            st.metric("Groundedness", f"{groundedness_color} {brief.groundedness.upper()}")
        with col3:
            st.metric("Sources", len(brief.citations))
        
        if brief.warnings:
            for warning in brief.warnings:
                st.warning(f"⚠️ {warning}")
        
        st.markdown("### Summary")
        st.markdown(brief.summary)
        
        if brief.key_insights:
            st.markdown("### Key Insights")
            for insight in brief.key_insights:
                st.markdown(f"- {insight}")
        
        st.markdown("### Sentiment Overview")
        st.info(brief.sentiment_overview)
        
        st.markdown("### 📚 Citations")
        for i, citation in enumerate(brief.citations, 1):
            st.markdown(f"{i}. [{citation.subreddit}]({citation.url}) - *{citation.text_snippet[:100]}...* (relevance: {citation.relevance_score:.1%})")


def render_reasoning_mode(tools, show_trace):
    """Render the full reasoning trace interface."""
    st.header("🔬 Full Socratic Reasoning")
    
    st.markdown("""
    This mode demonstrates the **Self-Socratic Refinement** loop:
    1. **PLAN** - Break down the question into sub-questions
    2. **EXECUTE** - Gather evidence using available tools
    3. **VERIFY** - Check for gaps and contradictions
    4. **FINALIZE** - Produce grounded answer or refuse
    """)
    
    question = st.text_area(
        "Your Question",
        placeholder="e.g., What are people saying about Tesla on wallstreetbets?",
        height=100
    )
    
    if st.button("🔬 Run Socratic Refinement", type="primary"):
        if not question:
            st.warning("Please enter a question")
            return
        
        with st.spinner("Running Socratic refinement loop..."):
            from src.reasoning.socratic_refine import SocraticRefiner
            
            refiner = SocraticRefiner(tools._retriever)
            result = refiner.refine(question)
        
        if result.refused:
            st.error(f"⛔ **Response Refused**")
            st.markdown(result.refusal_reason)
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Confidence", f"{result.confidence:.1%}")
            with col2:
                st.metric("Groundedness", f"{result.groundedness_score:.1%}")
            with col3:
                verified = result.verification.claims_verified
                rejected = result.verification.claims_rejected
                st.metric("Verification", f"{verified}/{verified+rejected} claims")
            
            st.markdown("### 💬 Answer")
            st.markdown(result.answer)
            
            st.markdown("### 📚 Citations")
            for citation in result.citations[:5]:
                st.markdown(f"- [{citation.subreddit}]({citation.url}) (relevance: {citation.relevance_score:.1%})")
        
        if show_trace:
            st.markdown("---")
            st.markdown("### 🔍 Reasoning Trace")
            
            trace = result.trace
            for entry in trace:
                stage_emoji = {
                    "planning": "📋",
                    "executing": "⚡",
                    "verifying": "✅",
                    "finalizing": "📝",
                    "complete": "🎉",
                    "refused": "⛔"
                }.get(entry["stage"], "•")
                
                with st.expander(f"{stage_emoji} **{entry['stage'].upper()}** - {entry['event']}", expanded=False):
                    st.json(entry["data"])
            
            if result.verification.gaps_found:
                st.markdown("#### ⚠️ Identified Gaps")
                for gap in result.verification.gaps_found:
                    st.warning(gap)


if __name__ == "__main__":
    main()
