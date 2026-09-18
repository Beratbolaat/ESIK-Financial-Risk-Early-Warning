"""Presentation layer for EŞİK. No model, scoring, or network operations."""
import base64
from functools import lru_cache
from html import escape
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent


def apply_visual_theme():
    st.html(ROOT / "assets" / "esik.css")


@lru_cache(maxsize=4)
def _image_uri(name):
    # Only application-owned assets are read. No user-supplied path or HTML.
    paths = {"inspect": "inspect.png", "voice": "voice.png"}
    path = ROOT / "assets" / paths[name]
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def sidebar_brand():
    st.html('''<div class="esik-brand"><div class="esik-wordmark">EŞİK<span>.</span></div>
        <p>Finansal riskte<br>inceleme önceliği</p></div>''')


def page_header(title, description, *, section, illustration=None, hero=False):
    art = ""
    if illustration:
        art = f'<img class="esik-art" src="{_image_uri(illustration)}" alt="" aria-hidden="true">'
    size = " esik-hero" if hero else ""
    image_class = " esik-with-art" if illustration else ""
    st.html(f'''<section class="esik-page-header{size}{image_class}">
        <div class="esik-header-copy"><p class="esik-eyebrow">{escape(section)}</p>
        <h1>{escape(title)}</h1><p class="esik-description">{escape(description)}</p></div>{art}
        </section>''')


def go_to_page(page):
    st.session_state["esik_page"] = page


def open_company(company_id):
    st.session_state["esik_company_pending"] = str(company_id)
    go_to_page("Şirket Detayı")


def company_navigation():
    st.html('''<nav class="esik-case-nav" aria-label="Şirket inceleme bölümleri">
        <a href="#finansal-kanit">01 · Finansal kanıt</a>
        <a href="#senaryo-analizi">02 · Senaryo analizi</a>
        <a href="#rapor-paylasimi">03 · Rapor paylaşımı</a></nav>''')


def outcome(captured, total, reviewed, precision):
    isabet = f"{precision * 100:.2f}".replace(".", ",")
    st.html(f'''<section class="esik-outcome" aria-label="Tarihsel test sonucu">
        <div class="esik-outcome-number">{int(captured)}<span> / {int(total)}</span></div>
        <div><strong>İlk {int(reviewed)} kayıtta yakalanan iflas</strong>
        <p>Tarihsel test sonucu. İnceleme listesindeki isabet %{isabet}.</p></div>
        <span class="esik-outcome-note">Sabit kapasite<br>İlk %10</span></section>''')


def context_strip():
    st.html('''<div class="esik-context"><span>ANALİST ÇALIŞMA ALANI</span>
        <span>Tarihsel veri <i></i> 1 yıllık tahmin ufku</span></div>''')
