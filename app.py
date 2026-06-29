"""
fejlesztési ötletek:
- időintervallumok a programokhoz, mi mennyi időt igényel
- lehetőség arra, hogy a felhasználó kiválassza, melyik forrásokat szeretné felhasználni az útitervhez
- chates felület, ahol a felhasználó kérdezhet, és a rendszer válaszolhat
- csevegés mentésének lehetősége, hogy a felhasználó később visszanézhesse a beszélgetést
- lehetőségek felajánlása csetelésnél, gyorsgombokkal (pl. "Mutass több éttermet", "Mutass több látnivalót", "Mutass több tippet", "Autókölcsönzési lehetőségek", "Mutass szállást")
"""

import os
from typing import List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import requests
from bs4 import BeautifulSoup
import streamlit as st

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper

# =====================================================================
# 1. KÖRNYEZETI BEÁLLÍTÁSOK ÉS OLDAL CONFIG
# =====================================================================
load_dotenv()

st.set_page_config(
    page_title="AI Utazási Tanácsadó",
    page_icon="🗺️",
    layout="centered"
)

if not os.getenv("GEMINI_API_KEY"):
    st.error("Hiba: A GEMINI_API_KEY nem található a környezeti változók között!")

# =====================================================================
# 2. ADATSTRUKTÚRÁK (PYDANTIC MODELLEK)
# =====================================================================
class SearchQueries(BaseModel):
    keywords: List[str] = Field(description="10 releváns keresési kulcsszó vagy kifejezés")
    time_limit: Optional[str] = Field(description="A felhasználó által megadott időkorlát, ha van (pl. hétvége, 3 nap)")

class SelectedSource(BaseModel):
    title: str = Field(description="A weboldal címe")
    link: str = Field(description="A weboldal pontos URL címe")
    category: str = Field(description="A találat témája alapján javasolt kategória")

class FilteredLinks(BaseModel):
    selected_sources: List[SelectedSource] = Field(description="A legrelevánsabb 3-4 kiválasztott forrás listája")

# =====================================================================
# 3. SEGÉDFÜGGVÉNYEK
# =====================================================================
def get_expanded_search_results(keywords: list, max_results: int = 5) -> str:
    search = DuckDuckGoSearchAPIWrapper()
    search_query = " ".join(keywords[:3])
    results = search.results(search_query, max_results=max_results)
    
    formatted_results = ""
    for i, res in enumerate(results, 1):
        formatted_results += f"{i}. Találat:\nCím: {res['title']}\nLink: {res['link']}\nLeírás: {res['snippet']}\n\n"
    return formatted_results

def scrape_full_text(url: str) -> str:
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        for script in soup(["script", "style"]):
            script.decompose()
            
        text = soup.get_text(separator=" ")
        lines = [line.strip() for line in text.splitlines()]
        return " ".join(p for p in lines if p)
    except Exception as e:
        return f"Hiba a következő oldal letöltésekor ({url}): {e}"

# =====================================================================
# 4. LLM ÉS LÁNCOK INICIALIZÁLÁSA
# =====================================================================
model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite", 
    temperature=0.0,
    api_key=os.getenv("GEMINI_API_KEY")
)

# 1. fázis: Kulcsszavak
parser1 = JsonOutputParser(pydantic_object=SearchQueries)
prompt_template1 = PromptTemplate(
    template="A felhasználó a következő kéréssel fordult hozzád: \"{user_input}\". Generálj 5-10 kulcsszót és keresd meg az időkorlátot!\n\n{format_instructions}",
    input_variables=["user_input"],
    partial_variables={"format_instructions": parser1.get_format_instructions()},
)
chain1 = prompt_template1 | model | parser1

# 2. fázis: Szűrés
parser2 = JsonOutputParser(pydantic_object=FilteredLinks)
prompt_template2 = PromptTemplate(
    template="Felhasználó kérése: \"{user_input}\"\nTalálatok:\n{kereses_eredmeny}\n\nVálaszd ki a legjobb 3-4 forrást és kategorizáld őket!\n\n{format_instructions}",
    input_variables=["user_input", "kereses_eredmeny"],
    partial_variables={"format_instructions": parser2.get_format_instructions()},
)
chain2 = prompt_template2 | model | parser2

# 3. fázis: Útiterv
prompt_template3 = PromptTemplate(
    template="""Egy profi utazási tanácsadó vagy. Felhasználó kérése: "{user_input}"
    
    Készíts részletes útitervet napokra és napszakokra bontva, kizárólag a lenti források alapján!
    
    Források:
    {lekapart_adatok}
    
    Használj szép Markdown formázást címsorokkal és listákkal!""",
    input_variables=["user_input", "lekapart_adatok"]
)
chain3 = prompt_template3 | model

# =====================================================================
# 5. STREAMLIT FELHASZNÁLÓI FELÜLET (UI)
# =====================================================================
st.title("🗺️ AI Utazási Tanácsadó & Ügynök")
st.write("Írd be, hova szeretnél utazni, és a rendszer megkeresi a legfrissebb tippeket az interneten!")

# Felhasználói input
user_input = st.text_input(
    label="Milyen utazást tervezel?",
    placeholder="Pl.: Szeretnék egy 3 napos utazást tervezni Budapestre, éttermekkel és látnivalókkal."
)

if st.button("Útiterv generálása 🚀", type="primary"):
    if not user_input.strip():
        st.warning("Kérlek, írj be egy érvényes kérést!")
    else:
        # A teljes folyamat futtatása vizuális visszajelzésekkel
        with st.spinner("🧠 1. Fázis: Kulcsszavak generálása..."):
            teszt = chain1.invoke({"user_input": user_input})
            
        with st.spinner("🔍 2. Fázis: Élő internetes keresés..."):
            kereses_eredmeny = get_expanded_search_results(teszt['keywords'], max_results=5)
            
        with st.spinner("🎯 3. Fázis: Legjobb források kiválasztása..."):
            szurt_talalatok = chain2.invoke({"user_input": user_input, "kereses_eredmeny": kereses_eredmeny})
            
        with st.spinner("🌐 4. Fázis: Weboldalak tartalmának lekaparása..."):
            weboldalak_tartalma = []
            for i in range(len(szurt_talalatok['selected_sources'])):
                url = szurt_talalatok['selected_sources'][i]['link']
                weboldalak_tartalma.append(scrape_full_text(url))
                
        with st.spinner("📝 5. Fázis: Személyre szabott útiterv összeállítása..."):
            lekapart_adatok = "\n\n".join(weboldalak_tartalma)
            utiterv = chain3.invoke({"user_input": user_input, "lekapart_adatok": lekapart_adatok})

        # --- EREDMÉNYEK MEGJELENÍTÉSE ---
        st.success("Az útiterv sikeresen elkészült!")
        
        # 1. Az útiterv kiírása
        st.markdown("---")
        st.markdown(utiterv.content[0]["text"])
        st.markdown("---")
        
        # 2. A felhasznált linkek elegáns megjelenítése
        st.subheader("🔗 Felhasznált források és ötletek:")
        for forras in szurt_talalatok['selected_sources']:
            # Streamlit link kártyák/gombok formájában
            st.markdown(f"**[{forras['category'].upper()}]** [{forras['title']}]({forras['link']})")

