#!/usr/bin/env python3
# secure_streamlit_football_bot.py - Sichere Version für Online-Deployment

import streamlit as st
import pandas as pd
import anthropic
from Statistik import SmartFootballAnalyzer

@st.cache_resource
def load_analyzer():
    """Analyzer mit sicherem API-Key laden"""
    # API Key aus Streamlit Secrets oder Environment
    try:
        api_key = st.secrets["ANTHROPIC_API_KEY"]
    except:
        api_key = st.text_input("Claude API Key", type="password")
        if not api_key:
            st.warning("Bitte API Key eingeben")
            st.stop()
    
    analyzer = SmartFootballAnalyzer()
    analyzer.api_key = api_key  # Key setzen
    
    if not analyzer.load_data():
        st.error("CSV-Daten nicht gefunden!")
        return None
    if not analyzer.setup_claude():
        st.error("Claude API Problem!")
        return None
    return analyzer

def main():
    st.set_page_config(
        page_title="⚽ Team Stats Bot", 
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.title("⚽ Fußball-Team Statistiken")
    st.markdown("*Intelligenter Chatbot für unsere Mannschaftsstatistiken*")
    
    # Analyzer laden
    analyzer = load_analyzer()
    if analyzer is None:
        st.stop()
    
    # Info über verfügbare Daten
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Spieler", len(analyzer.all_players))
    with col2:
        st.metric("Datensätze", len(analyzer.df))
    with col3:
        st.metric("Spiele", analyzer.df['event_id'].nunique())
    with col4:
        st.metric("Tore gesamt", analyzer.df['goals'].sum())
    
    st.markdown("---")
    
    # Chat History initialisieren
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hallo! Stellen Sie mir beliebige Fragen über die Mannschaftsstatistiken. Ich kann komplexe Analysen durchführen!"}
        ]
    
    # Sidebar mit Quick-Actions
    with st.sidebar:
        st.header("🚀 Quick-Fragen")
        
        quick_questions = [
            "Wer ist der beste Torschütze?",
            "Wer trifft in der ersten Halbzeit am häugisten",
            "Wer hat die meisten Vorlagen gemacht",
            "Mit wem gewinnt Jari am häufigsten?",
            "Alle Top-Scorer und Assisters"
        ]
        
        for q in quick_questions:
            if st.button(q, key=f"quick_{hash(q)}"):
                st.session_state.messages.append({"role": "user", "content": q})
                st.rerun()
        
        st.markdown("---")
        st.markdown("**💡 Beispiele:**")
        st.markdown("• *Siegrate von [Spieler]*")
        st.markdown("• *[Spieler] vs [Spieler]*") 
        st.markdown("• *Beste 4er-Formation*")
        st.markdown("• *Tore in Minute 30-40*")
    
    # Chat Interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # User Input
    if prompt := st.chat_input("Frage über Teamstatistiken..."):
        # User message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Bot response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Analysiere Daten..."):
                try:
                    context = analyzer.analyze_question(prompt)
                    answer = analyzer.ask_claude(prompt, context)
                    st.markdown(answer)
                    
                    # Bot message zur History
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    error_msg = f"Entschuldigung, es gab einen Fehler bei der Analyse: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    # Footer
    st.markdown("---")
    st.markdown("*Powered by Claude AI & Streamlit*")

if __name__ == "__main__":
    main()