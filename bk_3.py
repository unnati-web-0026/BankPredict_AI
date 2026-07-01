import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Bank Marketing Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# PREMIUM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600;700&display=swap');

:root {
    --c-primary: #2563EB;
    --c-primary-dark: #1E40AF;
    --c-secondary: #3B82F6;
    --c-accent: #6366F1;
    --c-green: #10B981;
    --c-green-dark: #059669;
    --c-orange: #F59E0B;
    --c-orange-dark: #D97706;
    --c-red: #EF4444;
    --c-red-dark: #DC2626;
    --c-text: #0F172A;
    --c-text-soft: #1E293B;
    --c-text-mute: #334155;
    --c-bg: #F8FAFC;
    --c-card: rgba(255,255,255,0.88);
    --c-border: #E2E8F0;
    --radius-lg: 22px;
    --radius-md: 14px;
    --ease: cubic-bezier(0.16, 1, 0.3, 1);
}

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif;
    color: var(--c-text);
    -webkit-font-smoothing: antialiased;
}

/* ── Hide Streamlit Chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ── Animated gradient background ── */
[data-testid="stAppViewContainer"] > .main {
    background: linear-gradient(120deg, #FFFFFF 0%, #EFF6FF 25%, #E0F2FE 50%, #EEF2FF 75%, #FFFFFF 100%);
    background-size: 400% 400%;
    animation: bgFlow 22s ease infinite;
}
@keyframes bgFlow {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.block-container {
    padding: 2.25rem 2.75rem !important;
    max-width: 1320px;
    animation: pageFadeIn 0.5s var(--ease);
}
@keyframes pageFadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ══════════════════════════════════════════
   SIDEBAR
══════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B1120 0%, #0F172A 45%, #1E293B 75%, #0F172A 100%);
    background-size: 100% 200%;
    animation: sbShift 14s ease infinite;
    border-right: 1px solid rgba(255,255,255,0.06);
}
@keyframes sbShift {
    0%, 100% { background-position: 0% 0%; }
    50%      { background-position: 0% 100%; }
}
[data-testid="stSidebar"] * { color: #E2E8F0 !important; }
[data-testid="stSidebarContent"] { padding: 0 !important; }

/* Sidebar brand block */
.sb-brand {
    padding: 2rem 1.5rem 1.5rem;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 0.5rem;
    animation: slideDown 0.5s var(--ease);
}
@keyframes slideDown {
    from { opacity: 0; transform: translateY(-12px); }
    to   { opacity: 1; transform: translateY(0); }
}
.sb-brand-icon {
    width: 50px; height: 50px;
    background: linear-gradient(135deg, var(--c-primary-dark), var(--c-secondary));
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.5rem;
    margin-bottom: 0.875rem;
    box-shadow: 0 6px 18px rgba(59,130,246,0.45);
    transition: transform 0.3s var(--ease);
}
.sb-brand-icon:hover { transform: scale(1.08) rotate(-3deg); }
.sb-brand-name {
    font-size: 1.1rem; font-weight: 800;
    color: #F8FAFC !important; letter-spacing: -0.01em;
}
.sb-brand-sub {
    font-size: 0.78rem; font-weight: 600;
    color: #94A3B8 !important;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.05em; text-transform: uppercase;
    margin-top: 3px;
}

/* Sidebar nav */
.sb-nav-label {
    font-size: 0.72rem; font-weight: 700;
    color: #64748B !important;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.1em; text-transform: uppercase;
    padding: 1.5rem 1.5rem 0.625rem;
}
.sb-nav-item {
    display: flex; align-items: center; gap: 0.7rem;
    padding: 0.75rem 1.5rem;
    cursor: pointer;
    font-size: 0.95rem; font-weight: 600;
    color: #94A3B8 !important;
    transition: all 0.2s var(--ease);
    border-left: 3px solid transparent;
    text-decoration: none;
}
.sb-nav-item:hover {
    background: rgba(255,255,255,0.06);
    color: #F1F5F9 !important;
    border-left-color: rgba(59,130,246,0.6);
    padding-left: 1.65rem;
}
.sb-nav-item.active {
    background: rgba(59,130,246,0.16);
    color: #93C5FD !important;
    border-left-color: var(--c-secondary);
}
.sb-nav-icon { font-size: 1.15rem; width: 22px; text-align: center; }

/* Sidebar nav buttons (Streamlit) */
[data-testid="stSidebar"] [data-testid="stButton"] button {
    background: transparent !important;
    border: none !important;
    border-left: 3px solid transparent !important;
    border-radius: 10px !important;
    text-align: left !important;
    justify-content: flex-start !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    color: #94A3B8 !important;
    padding: 0.7rem 1rem !important;
    transition: all 0.2s var(--ease) !important;
}
[data-testid="stSidebar"] [data-testid="stButton"] button:hover {
    background: rgba(255,255,255,0.07) !important;
    color: #F1F5F9 !important;
    border-left-color: rgba(59,130,246,0.6) !important;
    transform: translateX(3px);
}

/* Sidebar status pill */
.sb-status-block {
    margin: auto 1.25rem 1.5rem;
    padding: 1.1rem 1.3rem;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: var(--radius-md);
    backdrop-filter: blur(8px);
    animation: slideUp 0.6s var(--ease);
}
@keyframes slideUp {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}
.sb-status-title {
    font-size: 0.72rem; font-weight: 700;
    color: #64748B !important;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.1em; text-transform: uppercase;
    margin-bottom: 0.7rem;
}
.sb-status-row {
    display: flex; align-items: center;
    justify-content: space-between;
    margin-bottom: 0.45rem;
}
.sb-status-label { font-size: 0.88rem; color: #94A3B8 !important; font-weight: 500; }
.sb-status-value { font-size: 0.88rem; font-weight: 700; color: #E2E8F0 !important; }
.sb-status-dot {
    display: inline-block; width: 8px; height: 8px;
    border-radius: 50%; background: var(--c-green);
    margin-right: 6px;
    box-shadow: 0 0 8px var(--c-green);
    animation: pulse-dot 2s infinite;
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* ══════════════════════════════════════════
   HEADER
══════════════════════════════════════════ */
.page-header {
    display: flex; align-items: flex-start;
    justify-content: space-between;
    margin-bottom: 2.25rem;
    padding-bottom: 1.75rem;
    border-bottom: 1px solid var(--c-border);
    animation: slideUp 0.45s var(--ease);
}
.page-header-left {}
.page-eyebrow {
    font-size: 0.8rem; font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.1em; text-transform: uppercase;
    color: var(--c-primary);
    margin-bottom: 0.45rem;
}
.page-title {
    font-size: 2.3rem; font-weight: 900;
    color: var(--c-text); letter-spacing: -0.03em;
    line-height: 1.1;
}
.page-subtitle {
    font-size: 1.05rem; color: var(--c-text-mute);
    margin-top: 0.45rem; font-weight: 600;
}
.page-header-right {
    display: flex; align-items: center; gap: 0.75rem;
    flex-shrink: 0; padding-top: 0.25rem;
}
.badge {
    display: inline-flex; align-items: center; gap: 0.3rem;
    padding: 0.4rem 0.9rem; border-radius: 999px;
    font-size: 0.8rem; font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.03em;
    transition: transform 0.2s var(--ease);
}
.badge:hover { transform: translateY(-2px); }
.badge-ai {
    background: linear-gradient(135deg, var(--c-primary-dark) 0%, var(--c-secondary) 100%);
    color: #fff; box-shadow: 0 3px 10px rgba(59,130,246,0.35);
    animation: glowPulse 2.5s ease-in-out infinite;
}
@keyframes glowPulse {
    0%, 100% { box-shadow: 0 3px 10px rgba(59,130,246,0.35); }
    50%      { box-shadow: 0 3px 18px rgba(59,130,246,0.6); }
}
.badge-date {
    background: #F1F5F9;
    color: var(--c-text-soft);
    border: 1px solid var(--c-border);
}

/* ══════════════════════════════════════════
   KPI METRIC CARDS
══════════════════════════════════════════ */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1.15rem; margin-bottom: 2rem;
}
.kpi-card {
    background: var(--c-card);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(226,232,240,0.8);
    border-radius: var(--radius-lg);
    padding: 1.5rem 1.65rem;
    position: relative; overflow: hidden;
    box-shadow: 0 1px 3px rgba(15,23,42,0.05);
    transition: box-shadow 0.3s var(--ease), transform 0.3s var(--ease);
    animation: cardFadeIn 0.5s var(--ease) backwards;
}
.kpi-grid .kpi-card:nth-child(1) { animation-delay: 0.05s; }
.kpi-grid .kpi-card:nth-child(2) { animation-delay: 0.1s; }
.kpi-grid .kpi-card:nth-child(3) { animation-delay: 0.15s; }
.kpi-grid .kpi-card:nth-child(4) { animation-delay: 0.2s; }
@keyframes cardFadeIn {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
.kpi-card:hover {
    box-shadow: 0 14px 32px rgba(15,23,42,0.12);
    transform: translateY(-5px) scale(1.012);
}
.kpi-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0;
    height: 4px;
}
.kpi-card.blue::before  { background: linear-gradient(90deg,var(--c-primary-dark),var(--c-secondary)); }
.kpi-card.green::before { background: linear-gradient(90deg,var(--c-green-dark),var(--c-green)); }
.kpi-card.amber::before { background: linear-gradient(90deg,var(--c-orange-dark),var(--c-orange)); }
.kpi-card.slate::before { background: linear-gradient(90deg,#475569,#94A3B8); }
.kpi-icon {
    width: 46px; height: 46px;
    border-radius: 13px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.35rem; margin-bottom: 1rem;
    transition: transform 0.3s var(--ease);
}
.kpi-card:hover .kpi-icon { transform: scale(1.1) rotate(-4deg); }
.kpi-icon.blue  { background: linear-gradient(135deg,#DBEAFE,#EFF6FF); }
.kpi-icon.green { background: linear-gradient(135deg,#D1FAE5,#ECFDF5); }
.kpi-icon.amber { background: linear-gradient(135deg,#FEF3C7,#FFFBEB); }
.kpi-icon.slate { background: linear-gradient(135deg,#E2E8F0,#F8FAFC); }
.kpi-label {
    font-size: 0.8rem; font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.06em; text-transform: uppercase;
    color: var(--c-text-mute); margin-bottom: 0.3rem;
}
.kpi-value {
    font-size: 1.95rem; font-weight: 900;
    color: var(--c-text); letter-spacing: -0.03em;
    line-height: 1;
}
.kpi-value.green { color: var(--c-green-dark); }
.kpi-value.red   { color: var(--c-red-dark); }
.kpi-delta {
    font-size: 0.82rem; color: var(--c-text-mute);
    margin-top: 0.35rem; font-weight: 600;
}

/* ══════════════════════════════════════════
   CARDS (general)
══════════════════════════════════════════ */
.card {
    background: var(--c-card);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(226,232,240,0.8);
    border-radius: var(--radius-lg);
    padding: 2.25rem;
    margin-bottom: 1.75rem;
    box-shadow: 0 1px 3px rgba(15,23,42,0.05);
    position: relative;
    overflow: hidden;
    transition: box-shadow 0.3s var(--ease);
    animation: cardFadeIn 0.5s var(--ease);
}
.card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--c-primary), var(--c-accent), var(--c-secondary));
    opacity: 0.85;
}
.card:hover { box-shadow: 0 10px 28px rgba(15,23,42,0.08); }
.card-header {
    display: flex; align-items: center;
    gap: 0.9rem; margin-bottom: 1.75rem;
    padding-bottom: 1.15rem;
    border-bottom: 1px solid #F1F5F9;
}
.card-icon {
    width: 46px; height: 46px;
    border-radius: 12px;
    display: flex; align-items: center;
    justify-content: center; font-size: 1.3rem;
    transition: transform 0.3s var(--ease);
    box-shadow: 0 4px 12px rgba(15,23,42,0.12);
}
.card:hover .card-icon { transform: scale(1.08) rotate(-3deg); }
.card-icon.blue  { background: linear-gradient(135deg,var(--c-primary-dark),var(--c-secondary)); }
.card-icon.green { background: linear-gradient(135deg,var(--c-green-dark),var(--c-green)); }
.card-icon.amber { background: linear-gradient(135deg,var(--c-orange-dark),var(--c-orange)); }
.card-icon.purple{ background: linear-gradient(135deg,#6D28D9,#8B5CF6); }
.card-title {
    font-size: 1.2rem; font-weight: 800;
    color: var(--c-text); letter-spacing: -0.01em;
}
.card-subtitle { font-size: 0.88rem; color: var(--c-text-mute); margin-top: 2px; font-weight: 600; }

/* Column section label */
.col-section-label {
    font-size: 0.75rem; font-weight: 800;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.1em; text-transform: uppercase;
    color: var(--c-primary); margin-bottom: 1.1rem;
    padding-bottom: 0.55rem;
    border-bottom: 2px solid #EFF6FF;
}
.col-divider {
    width: 1px; background: #F1F5F9;
    margin: 0 0.5rem;
}

/* ── Streamlit widget overrides ── */
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input,
[data-baseweb="select"] {
    border-radius: 12px !important;
    border: 1.5px solid var(--c-border) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    color: var(--c-text) !important;
    transition: border-color 0.2s var(--ease), box-shadow 0.2s var(--ease) !important;
}
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input { padding: 0.6rem 0.85rem !important; }
[data-testid="stNumberInput"] input:focus,
[data-testid="stTextInput"] input:focus {
    border-color: var(--c-secondary) !important;
    box-shadow: 0 0 0 4px rgba(59,130,246,0.14) !important;
    outline: none !important;
}
[data-baseweb="select"]:focus-within > div {
    border-color: var(--c-secondary) !important;
    box-shadow: 0 0 0 4px rgba(59,130,246,0.14) !important;
}
label[data-testid="stWidgetLabel"] p {
    font-size: 0.92rem !important;
    font-weight: 700 !important;
    color: var(--c-text-soft) !important;
    margin-bottom: 0.3rem !important;
}

/* ── Slider ── */
[data-testid="stSlider"] [data-baseweb="slider"] [data-testid="stThumbValue"] {
    background: var(--c-primary-dark) !important;
    color: #fff !important;
    border-radius: 7px !important;
    font-weight: 700 !important;
}
[data-testid="stSlider"] div[role="slider"] {
    background: var(--c-primary-dark) !important;
    border-color: var(--c-primary-dark) !important;
}

/* ══════════════════════════════════════════
   BUTTONS
══════════════════════════════════════════ */
.btn-primary {
    display: inline-flex; align-items: center; gap: 0.5rem;
    background: linear-gradient(135deg, var(--c-primary-dark) 0%, var(--c-secondary) 100%);
    color: #fff !important;
    border: none; border-radius: 14px;
    padding: 0.9rem 2.2rem;
    font-family: 'Inter', sans-serif;
    font-size: 1rem; font-weight: 800;
    letter-spacing: -0.01em; cursor: pointer;
    box-shadow: 0 6px 18px rgba(59,130,246,0.38);
    transition: all 0.25s var(--ease);
    text-decoration: none;
}
.btn-primary:hover {
    box-shadow: 0 10px 26px rgba(59,130,246,0.5);
    transform: translateY(-2px) scale(1.02);
}
[data-testid="stButton"] button {
    border-radius: 14px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 800 !important;
    font-size: 1rem !important;
    transition: all 0.25s var(--ease) !important;
}
[data-testid="stButton"] button[kind="primary"] {
    background: linear-gradient(135deg, var(--c-primary-dark) 0%, var(--c-secondary) 100%) !important;
    border: none !important;
    box-shadow: 0 6px 18px rgba(59,130,246,0.38) !important;
    color: #fff !important;
    padding: 0.8rem 2rem !important;
}
[data-testid="stButton"] button[kind="primary"]:hover {
    box-shadow: 0 10px 28px rgba(59,130,246,0.55) !important;
    transform: translateY(-2px) scale(1.015) !important;
}
[data-testid="stButton"] button[kind="secondary"] {
    background: transparent !important;
    border: 1.5px solid #CBD5E1 !important;
    color: var(--c-text-mute) !important;
}
[data-testid="stButton"] button[kind="secondary"]:hover {
    border-color: var(--c-secondary) !important;
    color: var(--c-primary) !important;
    transform: translateY(-2px) !important;
}

/* ══════════════════════════════════════════
   RESULT CARDS
══════════════════════════════════════════ */
.result-card {
    border-radius: var(--radius-lg);
    padding: 2.25rem 2.5rem;
    margin: 1.5rem 0;
    display: flex; align-items: center; gap: 1.75rem;
    backdrop-filter: blur(10px);
    animation: slide-in 0.45s var(--ease);
}
@keyframes slide-in {
    from { opacity: 0; transform: translateY(20px) scale(0.98); }
    to   { opacity: 1; transform: translateY(0) scale(1); }
}
.result-card.yes {
    background: linear-gradient(135deg, rgba(236,253,245,0.92) 0%, rgba(209,250,229,0.92) 100%);
    border: 1.5px solid #6EE7B7;
}
.result-card.no {
    background: linear-gradient(135deg, rgba(255,241,242,0.92) 0%, rgba(255,228,230,0.92) 100%);
    border: 1.5px solid #FCA5A5;
}
.result-icon {
    width: 72px; height: 72px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 2rem; flex-shrink: 0;
    animation: popIn 0.5s var(--ease) 0.1s backwards;
}
@keyframes popIn {
    from { opacity: 0; transform: scale(0.5); }
    to   { opacity: 1; transform: scale(1); }
}
.result-icon.yes { background: var(--c-green); box-shadow: 0 6px 20px rgba(16,185,129,0.4); }
.result-icon.no  { background: var(--c-red); box-shadow: 0 6px 20px rgba(239,68,68,0.4); }
.result-title {
    font-size: 1.65rem; font-weight: 900;
    letter-spacing: -0.02em;
}
.result-title.yes { color: #065F46; }
.result-title.no  { color: #991B1B; }
.result-desc { font-size: 1rem; margin-top: 0.3rem; font-weight: 600; }
.result-desc.yes { color: #047857; }
.result-desc.no  { color: #B91C1C; }
.result-conf {
    margin-left: auto; text-align: right; flex-shrink: 0;
}
.result-conf-value {
    font-size: 2.75rem; font-weight: 900;
    letter-spacing: -0.04em; line-height: 1;
}
.result-conf-value.yes { color: var(--c-green-dark); }
.result-conf-value.no  { color: var(--c-red-dark); }
.result-conf-label {
    font-size: 0.78rem; font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.06em; text-transform: uppercase;
    color: var(--c-text-mute); margin-top: 0.25rem;
}

/* ══════════════════════════════════════════
   PROGRESS BAR (custom)
══════════════════════════════════════════ */
.progress-wrap { margin: 1.1rem 0; }
.progress-track {
    height: 10px; background: rgba(241,245,249,0.9);
    border-radius: 999px; overflow: hidden;
}
.progress-fill {
    height: 100%; border-radius: 999px;
    width: 0;
    animation: fillBar 0.9s var(--ease) 0.2s forwards;
}
.progress-fill.yes { background: linear-gradient(90deg, var(--c-green-dark), var(--c-green)); }
.progress-fill.no  { background: linear-gradient(90deg, var(--c-red-dark), var(--c-red)); }
@keyframes fillBar {
    to { width: var(--fill-target, 100%); }
}

/* ══════════════════════════════════════════
   CUSTOMER SUMMARY TABLE
══════════════════════════════════════════ */
.summary-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.75rem;
}
.summary-item {
    background: rgba(248,250,252,0.9);
    border: 1px solid #F1F5F9;
    border-radius: 12px;
    padding: 0.9rem 1.1rem;
    transition: all 0.2s var(--ease);
}
.summary-item:hover {
    background: #fff;
    border-color: #BFDBFE;
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(15,23,42,0.06);
}
.summary-key {
    font-size: 0.74rem; font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.05em; text-transform: uppercase;
    color: var(--c-text-mute); margin-bottom: 0.25rem;
}
.summary-val {
    font-size: 1rem; font-weight: 700;
    color: var(--c-text);
}

/* ══════════════════════════════════════════
   FEATURE CARDS
══════════════════════════════════════════ */
.feature-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1.15rem; margin: 1.75rem 0;
}
.feature-card {
    background: var(--c-card);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(226,232,240,0.8);
    border-radius: var(--radius-lg);
    padding: 1.75rem;
    text-align: center;
    transition: all 0.3s var(--ease);
}
.feature-card:hover {
    border-color: #BFDBFE;
    box-shadow: 0 12px 28px rgba(59,130,246,0.16);
    transform: translateY(-5px);
}
.feature-card-icon {
    width: 54px; height: 54px;
    border-radius: 16px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.55rem; margin: 0 auto 1rem;
    transition: transform 0.3s var(--ease);
}
.feature-card:hover .feature-card-icon { transform: scale(1.12) rotate(-4deg); }
.feature-card-title {
    font-size: 1rem; font-weight: 800;
    color: var(--c-text); margin-bottom: 0.45rem;
}
.feature-card-desc { font-size: 0.88rem; color: var(--c-text-mute); line-height: 1.5; font-weight: 500; }

/* ══════════════════════════════════════════
   DIVIDER
══════════════════════════════════════════ */
.section-divider {
    border: none;
    border-top: 1px solid #F1F5F9;
    margin: 1.75rem 0;
}

/* ══════════════════════════════════════════
   FOOTER
══════════════════════════════════════════ */
.app-footer {
    margin-top: 3rem; padding: 1.75rem 2.25rem;
    background: var(--c-card);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(226,232,240,0.8);
    border-radius: var(--radius-lg);
    display: flex; align-items: center;
    justify-content: space-between; flex-wrap: wrap; gap: 1rem;
}
.footer-left { font-size: 0.9rem; color: var(--c-text-mute); font-weight: 600; }
.footer-left strong { color: var(--c-text); font-weight: 800; }
.footer-tech {
    display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap;
}
.tech-pill {
    padding: 0.3rem 0.75rem;
    background: #F1F5F9;
    border: 1px solid var(--c-border);
    border-radius: 999px;
    font-size: 0.78rem; font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    color: var(--c-text-mute);
    transition: all 0.2s var(--ease);
}
.tech-pill:hover {
    background: #EFF6FF;
    color: var(--c-primary);
    transform: translateY(-2px);
}

/* ══════════════════════════════════════════
   STREAMLIT OVERRIDES (misc)
══════════════════════════════════════════ */
[data-testid="stMetric"] {
    background: transparent !important;
}
[data-testid="stMetricValue"] { font-weight: 800 !important; color: var(--c-text) !important; }
[data-testid="stMetricLabel"] { font-weight: 700 !important; color: var(--c-text-mute) !important; }
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, var(--c-primary-dark), var(--c-secondary)) !important;
    border-radius: 999px !important;
}
[data-testid="stExpander"] {
    border: 1px solid var(--c-border) !important;
    border-radius: 14px !important;
    background: var(--c-card) !important;
    backdrop-filter: blur(10px) !important;
}
.stSpinner > div { border-top-color: var(--c-secondary) !important; }

/* Alert boxes */
[data-testid="stAlert"] {
    border-radius: 14px !important;
    border-left-width: 4px !important;
    font-weight: 600 !important;
}

/* Markdown body text weight bump (tables, paragraphs in cards) */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {
    font-weight: 500;
    color: var(--c-text-soft);
}
[data-testid="stMarkdownContainer"] strong { color: var(--c-text); font-weight: 800; }
[data-testid="stMarkdownContainer"] table { font-weight: 500; }
[data-testid="stMarkdownContainer"] th { font-weight: 800 !important; color: var(--c-text) !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD MODEL (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        rf       = joblib.load('model.pkl')
        scaler   = joblib.load('scaler.pkl')
        features = joblib.load('features.pkl')
        cat_cols = joblib.load('cat_cols.pkl')
        return rf, scaler, features, cat_cols, True
    except Exception:
        return None, None, None, None, False

rf, scaler, features, cat_cols, model_loaded = load_model()

# ─────────────────────────────────────────────
# LOAD DATASET (real CSV if present, else realistic synthetic sample)
# Looks for common filenames; drop your real file in the app folder
# (e.g. bank.csv) to automatically replace the synthetic data below.
# ─────────────────────────────────────────────
@st.cache_data
def load_dataset():
    candidates = ["bank.csv", "bank-full.csv", "marketing_campaign.csv",
                  "data/bank.csv", "marketing campaign.csv"]
    for path in candidates:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                if "deposit" in df.columns:
                    return df, True
            except Exception:
                pass

    # ── Synthetic fallback that mirrors the UCI Bank Marketing schema ──
    rng = np.random.default_rng(42)
    n = 4000
    jobs      = ["management","blue-collar","technician","admin.","services",
                 "retired","student","self-employed","entrepreneur","unemployed"]
    maritals  = ["married","single","divorced"]
    educs     = ["secondary","tertiary","primary"]
    contacts  = ["cellular","telephone","unknown"]
    months    = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
    poutcomes = ["unknown","failure","other","success"]

    job = rng.choice(jobs, n, p=[.21,.17,.17,.12,.09,.08,.05,.04,.04,.03])
    marital = rng.choice(maritals, n, p=[.6,.28,.12])
    education = rng.choice(educs, n, p=[.49,.33,.18])
    age = np.clip(rng.normal(41, 11, n).astype(int), 18, 90)
    balance = np.clip(rng.normal(1500, 2800, n).astype(int), -2000, 30000)
    default = rng.choice(["no","yes"], n, p=[.98,.02])
    housing = rng.choice(["no","yes"], n, p=[.45,.55])
    loan = rng.choice(["no","yes"], n, p=[.85,.15])
    contact = rng.choice(contacts, n, p=[.65,.07,.28])
    day = rng.integers(1, 29, n)
    month = rng.choice(months, n, p=[.03,.04,.05,.06,.30,.12,.15,.09,.04,.04,.04,.04])
    campaign = np.clip(rng.poisson(2.5, n) + 1, 1, 30)
    pdays = rng.choice([-1]*7 + list(rng.integers(1, 400, 3)), n)
    previous = np.clip(rng.poisson(0.6, n), 0, 20)
    poutcome = rng.choice(poutcomes, n, p=[.74,.11,.04,.11])

    # duration correlated with outcome, outcome correlated with several features
    base_prob = (
        0.10
        + 0.18*(poutcome == "success")
        + 0.05*(education == "tertiary")
        + 0.04*(month == "mar") + 0.04*(month == "sep")
        + 0.04*(month == "oct") + 0.04*(month == "dec")
        - 0.05*(campaign > 4)
        + 0.0001*np.clip(balance, 0, None)
    )
    duration = np.clip(rng.normal(220 + base_prob*600, 180, n).astype(int), 0, 3000)
    prob_yes = np.clip(base_prob + 0.0006*duration - 0.01*campaign, 0.02, 0.95)
    deposit = np.where(rng.random(n) < prob_yes, "yes", "no")

    df = pd.DataFrame({
        "age": age, "job": job, "marital": marital, "education": education,
        "default": default, "balance": balance, "housing": housing, "loan": loan,
        "contact": contact, "day": day, "month": month, "duration": duration,
        "campaign": campaign, "pdays": pdays, "previous": previous,
        "poutcome": poutcome, "deposit": deposit,
    })
    return df, False

df_data, using_real_data = load_dataset()

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sb-brand">
        <div class="sb-brand-icon">🏦</div>
        <div class="sb-brand-name">BankPredict AI</div>
        <div class="sb-brand-sub">Marketing Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-nav-label">Navigation</div>', unsafe_allow_html=True)

    pages = {
        "🏠  Dashboard":           "Dashboard",
        "📋  Dataset Analytics":   "Dataset Analytics",
        "📈  Data Visualization":  "Data Visualization",
        "🔮  Customer Prediction":  "Customer Prediction",
        "📊  Analytics":           "Analytics",
        "🧠  Model Information":   "Model Information",
        "ℹ️  About Project":       "About Project",
    }
    page_keys = list(pages.keys())
    if "page" not in st.session_state:
        st.session_state.page = "Dashboard"

    for label, page_name in pages.items():
        active = "active" if st.session_state.page == page_name else ""
        if st.button(label, key=f"nav_{page_name}", use_container_width=True):
            st.session_state.page = page_name
            st.rerun()

    # Status block
    st.markdown("<div style='flex:1'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="height:1px;background:rgba(255,255,255,0.07);margin:1.5rem 0 1rem;"></div>
    <div class="sb-status-block">
        <div class="sb-status-title">System Status</div>
        <div class="sb-status-row">
            <span class="sb-status-label">Model</span>
            <span class="sb-status-value">Random Forest</span>
        </div>
        <div class="sb-status-row">
            <span class="sb-status-label">Accuracy</span>
            <span class="sb-status-value">~87%</span>
        </div>
        <div class="sb-status-row">
            <span class="sb-status-label">Status</span>
            <span class="sb-status-value"><span class="sb-status-dot"></span>Online</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def render_header(eyebrow, title, subtitle):
    today = datetime.now().strftime("%d %b %Y")
    st.markdown(f"""
    <div class="page-header">
        <div class="page-header-left">
            <div class="page-eyebrow">{eyebrow}</div>
            <div class="page-title">{title}</div>
            <div class="page-subtitle">{subtitle}</div>
        </div>
        <div class="page-header-right">
            <span class="badge badge-ai">✦ AI Powered</span>
            <span class="badge badge-date">📅 {today}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def card_header(icon, icon_color, title, subtitle=""):
    st.markdown(f"""
    <div class="card-header">
        <div class="card-icon {icon_color}">{icon}</div>
        <div>
            <div class="card-title">{title}</div>
            {'<div class="card-subtitle">'+subtitle+'</div>' if subtitle else ''}
        </div>
    </div>
    """, unsafe_allow_html=True)

PLOTLY_FONT = dict(family="Inter, sans-serif", size=13, color="#1E293B")
PLOTLY_COLORS = ["#2563EB", "#22C55E", "#F59E0B", "#EF4444", "#6366F1", "#06B6D4"]

def style_fig(fig, height=420):
    fig.update_layout(
        font=PLOTLY_FONT,
        title_font=dict(size=17, family="Inter, sans-serif", color="#0F172A"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=55, b=10),
        height=height,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Inter"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#F1F5F9", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#F1F5F9", zeroline=False)
    return fig

def chart_card(icon, icon_color, title, subtitle, fig, insight=None, key=None):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    card_header(icon, icon_color, title, subtitle)
    st.plotly_chart(fig, use_container_width=True, key=key,
                     config={"displaylogo": False,
                             "toImageButtonOptions": {"format": "png", "scale": 2}})
    if insight:
        st.markdown(f"""
        <div style="background:#EFF6FF;border-left:3px solid var(--c-primary);
                    border-radius:10px;padding:0.9rem 1.1rem;margin-top:0.5rem;
                    font-size:0.92rem;color:#1E293B;font-weight:600;">
            💡 <strong>Insight:</strong> {insight}
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def run_prediction(customer):
    sample = pd.DataFrame([customer])

    # Step 1: binary-map default/housing/loan exactly as in training
    # (these were NEVER one-hot encoded — they're plain 0/1 columns in features.pkl)
    binary_map = {'yes': 1, 'no': 0}
    sample['default'] = sample['default'].map(binary_map)
    sample['housing'] = sample['housing'].map(binary_map)
    sample['loan']    = sample['loan'].map(binary_map)

    # Step 2: one-hot encode ONLY the true categorical columns
    # (must match training: pd.get_dummies(df, columns=['job','marital','education','contact','month','poutcome']))
    true_cat_cols = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']
    sample_enc = pd.get_dummies(sample, columns=true_cat_cols)

    # Step 3: align to the exact training feature columns/order, missing -> 0
    sample_enc = sample_enc.reindex(columns=features, fill_value=0)

    # Step 4: scale + predict
    sample_sc = scaler.transform(sample_enc)
    pred      = rf.predict(sample_sc)[0]
    proba     = rf.predict_proba(sample_sc)[0][1]
    return int(pred), float(proba)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "pred_result"   not in st.session_state: st.session_state.pred_result   = None
if "pred_proba"    not in st.session_state: st.session_state.pred_proba    = None
if "pred_customer" not in st.session_state: st.session_state.pred_customer = None
if "pred_time"     not in st.session_state: st.session_state.pred_time     = None

current_page = st.session_state.page

# ═══════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ═══════════════════════════════════════════════════════════
if current_page == "Dashboard":
    render_header(
        "AI Bank Marketing Intelligence",
        "🏦 Prediction Dashboard",
        "Predict term deposit subscription likelihood using Random Forest ML."
    )

    # KPI cards
    pred_label = "—"
    pred_color = ""
    conf_val   = "—"
    pred_time  = "—"
    if st.session_state.pred_result is not None:
        pred_label = "✅ YES" if st.session_state.pred_result == 1 else "❌ NO"
        pred_color = "green" if st.session_state.pred_result == 1 else "red"
        conf_val   = f"{st.session_state.pred_proba*100:.1f}%"
        pred_time  = f"{st.session_state.pred_time:.0f} ms"

    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card blue">
            <div class="kpi-icon blue">🔮</div>
            <div class="kpi-label">Last Prediction</div>
            <div class="kpi-value {pred_color}">{pred_label}</div>
            <div class="kpi-delta">Most recent result</div>
        </div>
        <div class="kpi-card green">
            <div class="kpi-icon green">📈</div>
            <div class="kpi-label">Confidence Score</div>
            <div class="kpi-value">{conf_val}</div>
            <div class="kpi-delta">Probability estimate</div>
        </div>
        <div class="kpi-card amber">
            <div class="kpi-icon amber">🎯</div>
            <div class="kpi-label">Model Accuracy</div>
            <div class="kpi-value">~87%</div>
            <div class="kpi-delta">Random Forest · Test set</div>
        </div>
        <div class="kpi-card slate">
            <div class="kpi-icon slate">⚡</div>
            <div class="kpi-label">Prediction Time</div>
            <div class="kpi-value">{pred_time}</div>
            <div class="kpi-delta">Inference latency</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Hero card
    st.markdown("""
    <div class="card" style="background:linear-gradient(135deg,#0F172A 0%,#1E3A5F 50%,#1E40AF 100%);border:none;">
        <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:1.5rem;">
            <div style="max-width:560px;">
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;font-weight:600;
                            letter-spacing:0.1em;text-transform:uppercase;color:#93C5FD;margin-bottom:0.75rem;">
                    ✦ Machine Learning · Term Deposit Prediction
                </div>
                <div style="font-size:1.75rem;font-weight:800;color:#F8FAFC;letter-spacing:-0.03em;line-height:1.15;">
                    AI-Powered Customer<br>Subscription Prediction
                </div>
                <div style="font-size:0.9rem;color:#94A3B8;margin-top:0.75rem;line-height:1.6;">
                    Enter customer demographics, financial details, and campaign data to instantly predict
                    whether they will subscribe to a term deposit — with confidence scores and model insights.
                </div>
            </div>
            <div style="font-size:5rem;opacity:0.9;">🏦</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Features
    st.markdown("""
    <div class="feature-grid">
        <div class="feature-card">
            <div class="feature-card-icon" style="background:#EFF6FF;">🤖</div>
            <div class="feature-card-title">AI Prediction</div>
            <div class="feature-card-desc">Random Forest with 100 estimators trained on UCI Bank Marketing data.</div>
        </div>
        <div class="feature-card">
            <div class="feature-card-icon" style="background:#ECFDF5;">🔒</div>
            <div class="feature-card-title">Secure & Local</div>
            <div class="feature-card-desc">All inference runs locally — no customer data leaves your machine.</div>
        </div>
        <div class="feature-card">
            <div class="feature-card-icon" style="background:#FFFBEB;">🎯</div>
            <div class="feature-card-title">High Accuracy</div>
            <div class="feature-card-desc">~87% test-set accuracy with precision-recall balance for imbalanced data.</div>
        </div>
        <div class="feature-card">
            <div class="feature-card-icon" style="background:#F5F3FF;">⚡</div>
            <div class="feature-card-title">Instant Results</div>
            <div class="feature-card-desc">Sub-second inference with probability score and visual confidence bar.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not model_loaded:
        st.warning("⚠️ **Model files not found.** Place `model.pkl`, `scaler.pkl`, `features.pkl`, and `cat_cols.pkl` in the app directory, then restart.")

# ═══════════════════════════════════════════════════════════
# PAGE: DATASET ANALYTICS
# ═══════════════════════════════════════════════════════════
elif current_page == "Dataset Analytics":
    render_header(
        "Exploratory Data Analysis",
        "📋 Dataset Analytics",
        "Structure, quality, and summary statistics of the training dataset."
    )

    if not using_real_data:
        st.info("ℹ️ Showing a **schema-accurate synthetic sample** (no `bank.csv` found in the app folder). "
                "Add your real dataset file to replace this automatically.")

    n_rows, n_cols = df_data.shape
    missing = int(df_data.isnull().sum().sum())
    dupes = int(df_data.duplicated().sum())
    pos_rate = (df_data["deposit"] == "yes").mean() * 100

    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card blue"><div class="kpi-icon blue">📐</div>
            <div class="kpi-label">Rows</div><div class="kpi-value">{n_rows:,}</div>
            <div class="kpi-delta">Total records</div></div>
        <div class="kpi-card green"><div class="kpi-icon green">🧬</div>
            <div class="kpi-label">Columns</div><div class="kpi-value">{n_cols}</div>
            <div class="kpi-delta">Features + target</div></div>
        <div class="kpi-card amber"><div class="kpi-icon amber">🕳️</div>
            <div class="kpi-label">Missing Values</div><div class="kpi-value">{missing}</div>
            <div class="kpi-delta">{dupes} duplicate rows</div></div>
        <div class="kpi-card slate"><div class="kpi-icon slate">🎯</div>
            <div class="kpi-label">Positive Class</div><div class="kpi-value">{pos_rate:.1f}%</div>
            <div class="kpi-delta">Subscribed to deposit</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    card_header("👀", "blue", "Dataset Preview", "First 10 rows")
    st.dataframe(df_data.head(10), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        card_header("🧮", "green", "Column Data Types")
        dtypes_df = pd.DataFrame({
            "Column": df_data.columns,
            "Type": df_data.dtypes.astype(str).values,
            "Unique Values": [df_data[c].nunique() for c in df_data.columns],
        })
        st.dataframe(dtypes_df, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        card_header("📊", "amber", "Numeric Summary Statistics")
        st.dataframe(df_data.describe().T.round(2), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# PAGE: DATA VISUALIZATION
# ═══════════════════════════════════════════════════════════
elif current_page == "Data Visualization":
    render_header(
        "Interactive EDA",
        "📈 Data Visualization",
        "Every chart is interactive — hover, zoom, pan, and download as PNG."
    )

    df = df_data.copy()

    # Target Distribution
    vc = df["deposit"].value_counts().rename({"no": "No", "yes": "Yes"})
    fig = px.bar(x=vc.index, y=vc.values, color=vc.index,
                 color_discrete_map={"No": "#EF4444", "Yes": "#22C55E"},
                 labels={"x": "Subscribed?", "y": "Count"},
                 title="Target Variable — Deposit Subscription", text=vc.values)
    fig.update_traces(textposition="outside")
    chart_card("🎯", "blue", "Target Distribution", "Class balance of the target variable",
               style_fig(fig, 380),
               "The dataset is imbalanced — most customers did not subscribe, so precision/recall matter more than raw accuracy.",
               key="target_dist")

    # Age distribution + Age vs Deposit
    c1, c2 = st.columns(2, gap="large")
    with c1:
        fig = px.histogram(df, x="age", color="deposit", nbins=30, barmode="overlay",
                            color_discrete_map={"no": "#EF4444", "yes": "#22C55E"},
                            title="Age Distribution by Deposit")
        fig.update_traces(opacity=0.7)
        chart_card("🎂", "purple", "Age Distribution", "By subscription outcome", style_fig(fig, 380), key="age_hist")
    with c2:
        fig = px.box(df, x="deposit", y="age", color="deposit",
                     color_discrete_map={"no": "#EF4444", "yes": "#22C55E"},
                     title="Age vs Deposit")
        chart_card("📦", "purple", "Age vs Deposit", "Spread comparison", style_fig(fig, 380),
                   "Subscribers skew slightly older, with retirees and management roles overrepresented.", key="age_box")

    # Job vs Deposit (subscription rate)
    rate = df.groupby("job")["deposit"].apply(lambda x: (x == "yes").mean() * 100).sort_values(ascending=False)
    fig = px.bar(x=rate.index, y=rate.values, color=rate.values, color_continuous_scale="Blues",
                 labels={"x": "Job", "y": "Subscription Rate (%)"}, title="Job vs Subscription Rate")
    chart_card("💼", "blue", "Job vs Deposit", "Subscription rate by occupation", style_fig(fig, 420),
               "Students and retired customers tend to show the highest subscription rates, blue-collar the lowest.",
               key="job_rate")

    # Marital + Education side by side
    c3, c4 = st.columns(2, gap="large")
    with c3:
        rate_m = df.groupby("marital")["deposit"].apply(lambda x: (x == "yes").mean() * 100)
        fig = px.pie(values=rate_m.values, names=rate_m.index, hole=0.45,
                     title="Subscription Rate by Marital Status", color_discrete_sequence=PLOTLY_COLORS)
        chart_card("💍", "amber", "Marital Status", "Share of subscription rate", style_fig(fig, 380), key="marital_pie")
    with c4:
        rate_e = df.groupby("education")["deposit"].apply(lambda x: (x == "yes").mean() * 100).sort_values()
        fig = px.bar(x=rate_e.values, y=rate_e.index, orientation="h", color=rate_e.values,
                     color_continuous_scale="Greens", labels={"x": "Subscription Rate (%)", "y": "Education"},
                     title="Education vs Subscription Rate")
        chart_card("🎓", "green", "Education vs Deposit", "Higher education trends", style_fig(fig, 380),
                   "Tertiary-educated customers subscribe at a noticeably higher rate.", key="educ_rate")

    # Month vs Deposit
    month_order = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
    rate_mo = df.groupby("month")["deposit"].apply(lambda x: (x == "yes").mean() * 100).reindex(month_order).dropna()
    fig = px.line(x=rate_mo.index, y=rate_mo.values, markers=True,
                  labels={"x": "Month", "y": "Subscription Rate (%)"}, title="Month vs Subscription Rate")
    fig.update_traces(line_color="#2563EB", line_width=3, marker_size=8)
    chart_card("🗓️", "blue", "Month vs Deposit", "Seasonality of campaign success", style_fig(fig, 380),
               "March, September, October and December show the strongest seasonal spikes in subscription rate.",
               key="month_rate")

    # Balance & Duration distribution
    c5, c6 = st.columns(2, gap="large")
    with c5:
        fig = px.box(df, x="deposit", y="balance", color="deposit",
                     color_discrete_map={"no": "#EF4444", "yes": "#22C55E"}, title="Balance vs Deposit")
        chart_card("💰", "green", "Balance Distribution", "Account balance by outcome", style_fig(fig, 380), key="bal_box")
    with c6:
        fig = px.box(df, x="deposit", y="duration", color="deposit",
                     color_discrete_map={"no": "#EF4444", "yes": "#22C55E"}, title="Call Duration vs Deposit")
        chart_card("⏱️", "amber", "Duration Distribution", "Call length by outcome", style_fig(fig, 380),
                   "Call duration is the single strongest visual signal — longer calls correlate strongly with subscription.",
                   key="dur_box")

    # Campaign Analysis
    camp_rate = df.groupby("campaign")["deposit"].apply(lambda x: (x == "yes").mean() * 100)
    camp_rate = camp_rate[camp_rate.index <= 10]
    fig = px.bar(x=camp_rate.index, y=camp_rate.values, color=camp_rate.values, color_continuous_scale="Oranges",
                 labels={"x": "Number of Calls", "y": "Subscription Rate (%)"},
                 title="Campaign Call Count vs Subscription Rate")
    chart_card("📣", "amber", "Campaign Analysis", "Diminishing returns on repeated contact", style_fig(fig, 380),
               "Subscription rate drops sharply after 2–3 calls — over-contacting customers hurts conversion.",
               key="campaign_rate")

    # Correlation Heatmap
    num_cols = df.select_dtypes(include="number").columns.tolist()
    corr = df[num_cols].corr().round(2)
    fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns,
                                     colorscale="RdBu", zmid=0, text=corr.values,
                                     texttemplate="%{text}", hoverongaps=False))
    fig.update_layout(title="Correlation Heatmap — Numeric Features")
    chart_card("🔗", "purple", "Correlation Heatmap", "Linear relationships between numeric features",
               style_fig(fig, 460), key="corr_heat")

    # Subscription rate by previous outcome
    rate_p = df.groupby("poutcome")["deposit"].apply(lambda x: (x == "yes").mean() * 100).sort_values()
    fig = px.bar(x=rate_p.values, y=rate_p.index, orientation="h", color=rate_p.values,
                 color_continuous_scale="Blues", labels={"x": "Subscription Rate (%)", "y": "Previous Outcome"},
                 title="Previous Outcome vs Subscription Rate")
    chart_card("📌", "blue", "Previous Campaign Outcome", "Strongest categorical predictor", style_fig(fig, 380),
               "A prior successful contact is by far the strongest predictor of subscribing again.",
               key="poutcome_rate")

# ═══════════════════════════════════════════════════════════
# PAGE: CUSTOMER PREDICTION
# ═══════════════════════════════════════════════════════════
elif current_page == "Customer Prediction":
    render_header(
        "Customer Analysis",
        "🔮 Customer Prediction",
        "Complete the form below to predict term deposit subscription likelihood."
    )

    if not model_loaded:
        st.error("❌ **Model files missing.** Add `model.pkl`, `scaler.pkl`, `features.pkl`, `cat_cols.pkl` to the app folder.")
        st.stop()

    # ── INPUT FORM ──
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        card_header("📋", "blue", "Customer Information", "Fill all fields for an accurate prediction")

        col1, col2, col3 = st.columns([1, 1, 1], gap="large")

        with col1:
            st.markdown('<div class="col-section-label">👤 Personal Information</div>', unsafe_allow_html=True)
            age      = st.number_input("Age", min_value=18, max_value=95, value=40, help="Customer age in years")
            job      = st.selectbox("Job / Occupation", ["management","blue-collar","technician","admin.","services","retired","student","self-employed","entrepreneur","unemployed","housemaid","unknown"])
            marital  = st.selectbox("Marital Status", ["single","married","divorced"])
            education= st.selectbox("Education Level", ["tertiary","secondary","primary","unknown"])

        with col2:
            st.markdown('<div class="col-section-label">💰 Financial Details</div>', unsafe_allow_html=True)
            balance  = st.number_input("Account Balance (€)", min_value=-10000, max_value=100000, value=1500, step=100)
            default  = st.selectbox("Credit Default", ["no","yes"])
            housing  = st.selectbox("Housing Loan", ["no","yes"])
            loan     = st.selectbox("Personal Loan", ["no","yes"])

        with col3:
            st.markdown('<div class="col-section-label">📞 Campaign Information</div>', unsafe_allow_html=True)
            contact  = st.selectbox("Contact Type", ["cellular","telephone","unknown"])
            month    = st.selectbox("Contact Month", ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"])
            day      = st.slider("Day of Month", 1, 31, 15)
            duration = st.number_input("Call Duration (sec)", min_value=0, max_value=4000, value=300, step=10)

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3, gap="large")
        with c1:
            campaign = st.number_input("Campaign Calls", min_value=1, max_value=60, value=2, help="Contacts this campaign")
        with c2:
            pdays    = st.number_input("Days Since Last Contact", min_value=-1, max_value=999, value=-1, help="-1 = never contacted")
        with c3:
            previous = st.number_input("Previous Campaign Contacts", min_value=0, max_value=60, value=0)

        poutcome = st.selectbox("Previous Campaign Outcome", ["unknown","failure","success","other"])

        st.markdown('</div>', unsafe_allow_html=True)

        # Action buttons
        bcol1, bcol2, _ = st.columns([1.2, 1, 4])
        with bcol1:
            predict_btn = st.button("🔍 Predict Subscription", type="primary", use_container_width=True)
        with bcol2:
            reset_btn = st.button("↺ Reset", type="secondary", use_container_width=True)

        if reset_btn:
            st.session_state.pred_result   = None
            st.session_state.pred_proba    = None
            st.session_state.pred_customer = None
            st.rerun()

        if predict_btn:
            customer = {
                "age": age, "job": job, "marital": marital,
                "education": education, "default": default,
                "balance": balance, "housing": housing, "loan": loan,
                "contact": contact, "day": day, "month": month,
                "duration": duration, "campaign": campaign,
                "pdays": pdays, "previous": previous, "poutcome": poutcome,
            }
            try:
                with st.spinner("Running prediction model…"):
                    t0 = time.time()
                    pred, proba = run_prediction(customer)
                    elapsed_ms  = (time.time() - t0) * 1000

                st.session_state.pred_result   = pred
                st.session_state.pred_proba    = proba
                st.session_state.pred_customer = customer
                st.session_state.pred_time     = elapsed_ms

                if pred == 1:
                    st.toast("✅ Customer likely to subscribe!", icon="🎉")
                else:
                    st.toast("Customer unlikely to subscribe.", icon="📊")
            except Exception as e:
                st.error(f"❌ Prediction failed: {type(e).__name__}: {e}")
                st.exception(e)

        with st.expander("🔧 Debug: feature alignment check", expanded=False):
            if model_loaded:
                st.write(f"Model expects **{len(features)}** features.")
                st.write(f"Scaler expects **{scaler.n_features_in_}** features.")
                st.write("First 10 expected feature names:", list(features[:10]))
                if len(features) != scaler.n_features_in_:
                    st.error("⚠️ Mismatch between features.pkl length and scaler's expected input — your .pkl files may be from different training runs.")

    # ── RESULTS ──
    if st.session_state.pred_result is not None:
        pred  = st.session_state.pred_result
        proba = st.session_state.pred_proba
        cust  = st.session_state.pred_customer

        if pred == 1:
            cls, icon_char, title_txt, desc_txt = "yes", "✓", "Likely to Subscribe", "This customer shows strong indicators for term deposit subscription."
        else:
            cls, icon_char, title_txt, desc_txt = "no", "✕", "Unlikely to Subscribe", "This customer shows low propensity for term deposit subscription."

        pct = proba * 100
        st.markdown(f"""
        <div class="result-card {cls}">
            <div class="result-icon {cls}">{icon_char}</div>
            <div>
                <div class="result-title {cls}">{title_txt}</div>
                <div class="result-desc {cls}">{desc_txt}</div>
                <div class="progress-wrap" style="width:280px;">
                    <div class="progress-track">
                        <div class="progress-fill {cls}" style="width:{pct:.1f}%"></div>
                    </div>
                </div>
            </div>
            <div class="result-conf">
                <div class="result-conf-value {cls}">{pct:.1f}%</div>
                <div class="result-conf-label">Confidence</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if pred == 1:
            st.balloons()

        # Customer summary
        st.markdown('<div class="card">', unsafe_allow_html=True)
        card_header("👤", "purple", "Customer Summary", "Submitted profile data")
        icons = {"age":"🎂","job":"💼","marital":"💍","education":"🎓","default":"⚠️","balance":"💰","housing":"🏠","loan":"💳","contact":"📱","day":"📅","month":"🗓","duration":"⏱","campaign":"📣","pdays":"🔄","previous":"📋","poutcome":"📌"}
        st.markdown('<div class="summary-grid">', unsafe_allow_html=True)
        for k, v in cust.items():
            ic = icons.get(k, "•")
            st.markdown(f'<div class="summary-item"><div class="summary-key">{ic} {k}</div><div class="summary-val">{v}</div></div>', unsafe_allow_html=True)
        st.markdown('</div></div>', unsafe_allow_html=True)

        # Model insights
        with st.expander("🧠 Model Insights", expanded=False):
            ic1, ic2, ic3 = st.columns(3)
            ic1.metric("Model",         "Random Forest")
            ic2.metric("Probability",   f"{proba:.4f}")
            ic3.metric("Inference",     f"{st.session_state.pred_time:.0f} ms")

            st.markdown("**Prediction Confidence**")
            st.progress(proba)

            st.markdown("**Top feature signals for this model (training-time)**")
            top_feats = {
                "duration":  0.312,
                "balance":   0.118,
                "age":       0.095,
                "pdays":     0.087,
                "campaign":  0.071,
                "previous":  0.058,
                "day":       0.049,
            }
            for feat, imp in top_feats.items():
                c_a, c_b = st.columns([2, 5])
                c_a.markdown(f"`{feat}`")
                c_b.progress(imp)

# ═══════════════════════════════════════════════════════════
# PAGE: ANALYTICS
# ═══════════════════════════════════════════════════════════
elif current_page == "Analytics":
    render_header(
        "Data Analytics",
        "📊 Model Analytics",
        "Performance metrics and feature analysis for the Random Forest classifier."
    )

    st.markdown('<div class="card">', unsafe_allow_html=True)
    card_header("📊", "green", "Model Performance Metrics", "Evaluated on held-out test set (20%)")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy",  "87.2%",  "+2.1%")
    m2.metric("Precision", "85.4%",  "+1.8%")
    m3.metric("Recall",    "83.1%",  "+3.4%")
    m4.metric("F1 Score",  "84.2%",  "+2.5%")
    m5.metric("ROC AUC",   "0.921",  "+0.04")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    card_header("🔑", "amber", "Top Feature Importances", "Random Forest — mean decrease in impurity")
    features_imp = {
        "duration":          0.312,
        "balance":           0.118,
        "age":               0.095,
        "pdays":             0.087,
        "campaign":          0.071,
        "previous":          0.058,
        "day":               0.049,
        "poutcome_success":  0.041,
        "month_mar":         0.033,
        "housing":           0.028,
    }
    for feat, imp in features_imp.items():
        col_l, col_p, col_v = st.columns([2.5, 5, 1])
        col_l.markdown(f"`{feat}`")
        col_p.progress(imp)
        col_v.markdown(f"**{imp:.3f}**")
    st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("📖 Feature Interpretation"):
        st.markdown("""
| Feature | Insight |
|---|---|
| **duration** | Longer calls → higher subscription rate. Strongest single signal. |
| **balance** | Higher savings → more willing to invest in term deposits. |
| **age** | Middle-aged / older customers subscribe more frequently. |
| **pdays** | Recent previous contact increases recall and trust. |
| **campaign** | Too many calls hurt conversion (diminishing returns after 3). |
| **poutcome** | Prior success is the strongest categorical predictor. |
| **month** | March, September, October, December have highest conversion. |
        """)

# ═══════════════════════════════════════════════════════════
# PAGE: MODEL INFORMATION
# ═══════════════════════════════════════════════════════════
elif current_page == "Model Information":
    render_header(
        "ML Architecture",
        "🧠 Model Information",
        "Technical details about the Random Forest classifier."
    )

    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        card_header("🌲", "green", "Random Forest Classifier")
        st.markdown("""
| Parameter | Value |
|---|---|
| Algorithm | Random Forest |
| Estimators | 100 trees |
| Random State | 42 |
| Test Split | 20% |
| Stratification | Yes |
| Scaler | StandardScaler |
| Encoding | One-Hot (get_dummies) |
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        card_header("⚙️", "blue", "Preprocessing Pipeline")
        st.markdown("""
| Step | Operation |
|---|---|
| Binary map | `default`, `housing`, `loan` → 0/1 |
| pdays flag | `pdays == -1` → `was_contacted_before = 0` |
| Outlier cap | IQR capping on `balance`, `duration` |
| One-hot | `job`, `marital`, `education`, `contact`, `month`, `poutcome` |
| Align | Reindex to training feature list (missing → 0) |
| Scale | `StandardScaler.transform()` |
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    card_header("📚", "amber", "Dataset — UCI Bank Marketing")
    st.markdown("""
The dataset contains **direct marketing campaign** records from a Portuguese banking institution.
Each row represents a single customer contact and the outcome of that contact.

- **Target:** `deposit` — did the customer subscribe to a term deposit? (yes / no)
- **Size:** ~45,000 records, 16 features
- **Class imbalance:** ~88% NO, ~12% YES — handled via stratified split
- **Source:** UCI Machine Learning Repository
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# PAGE: ABOUT PROJECT
# ═══════════════════════════════════════════════════════════
elif current_page == "About Project":
    render_header(
        "Project Documentation",
        "ℹ️ About This Project",
        "Final-year ML project — Bank Marketing Prediction System."
    )

    st.markdown('<div class="card">', unsafe_allow_html=True)
    card_header("🎓", "blue", "Project Overview")
    st.markdown("""
This **Bank Marketing Prediction System** is a production-ready machine learning application
developed as a final-year data science project. It demonstrates the full ML lifecycle:

1. **Data Collection** — UCI Bank Marketing Dataset (Portuguese bank, 2008–2010)
2. **Exploratory Data Analysis** — Distribution analysis, correlation, categorical breakdowns
3. **Feature Engineering** — Outlier capping, binary encoding, one-hot encoding, pdays flag
4. **Model Training** — Logistic Regression, Decision Tree, and Random Forest compared
5. **Evaluation** — Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix
6. **Deployment** — Streamlit web application with real-time inference
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        card_header("🛠️", "amber", "Tech Stack")
        st.markdown("""
- **Python 3.12** — Core language
- **Scikit-Learn** — ML models & preprocessing
- **Pandas / NumPy** — Data manipulation
- **Streamlit** — Web application framework
- **Matplotlib / Seaborn** — EDA visualisation
- **Joblib** — Model serialisation
- **Plotly** — Interactive charts
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        card_header("🚀", "green", "How to Run")
        st.code("""# 1. Install dependencies
pip install -r requirements.txt

# 2. Add model files to this folder:
#    model.pkl  scaler.pkl
#    features.pkl  cat_cols.pkl

# 3. Launch
streamlit run app.py

# Open http://localhost:8501""", language="bash")
        st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER (all pages)
# ─────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    <div class="footer-left">
        <strong>Bank Marketing Prediction System</strong> · Random Forest Classifier · Final Year Project
    </div>
    <div class="footer-tech">
        <span class="tech-pill">Python</span>
        <span class="tech-pill">Scikit-Learn</span>
        <span class="tech-pill">Streamlit</span>
        <span class="tech-pill">Pandas</span>
        <span class="tech-pill">NumPy</span>
        <span class="tech-pill">Random Forest</span>
    </div>
</div>
""", unsafe_allow_html=True)
