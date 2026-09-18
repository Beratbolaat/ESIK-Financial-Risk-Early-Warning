"""Deterministic, evidence-based assessments; no historical test outcome in report inputs."""
import json
import unicodedata
import re

import numpy as np
import pandas as pd
from esik_calibration import company_probability


def observed_scenario_candidates(shap_rows, company_row, limit=3):
    """Only observed, finite financial inputs can be changed in What-if."""
    observed_values = shap_rows['VARIABLE'].map(pd.to_numeric(company_row, errors='coerce'))
    was_missing = shap_rows['WAS_MISSING'].astype(str).str.lower().eq('true')
    eligible = ~was_missing & observed_values.notna() & np.isfinite(observed_values)
    return shap_rows.loc[eligible].sort_values('ABS_SHAP_VALUE', ascending=False, kind='stable').head(limit)


def percent(value):
    if 0 < value < .0005:return '<%0,1'
    if .9995 < value < 1:return '>%99,9'
    return ('%'+f'{value*100:.1f}').replace('.',',')


def evidence_table(shap_rows, reference):
    selected = pd.concat([
        shap_rows.loc[shap_rows.SHAP_VALUE>0].nlargest(3,'ABS_SHAP_VALUE'),
        shap_rows.loc[shap_rows.SHAP_VALUE<0].nlargest(2,'ABS_SHAP_VALUE')])
    rows=[]
    for r in selected.itertuples():
        missing=str(r.WAS_MISSING).lower()=='true';ref=reference.get(r.VARIABLE,{})
        rows.append({'Değişken':r.VARIABLE,'Finansal gösterge':r.DESCRIPTION,
            'Gözlenen değer':None if missing else float(r.MODEL_INPUT_VALUE),
            'Modelin kullandığı değer':float(r.MODEL_INPUT_VALUE),
            'Geliştirme medyanı':ref.get('median'),
            'Girdi kaynağı':'Eksik; eğitim medyanıyla dolduruldu' if missing else 'Gözlenen girdi',
            'Model etkisi':'Skoru artırıyor' if r.SHAP_VALUE>0 else 'Skoru azaltıyor',
            'SHAP (log-odds)':float(r.SHAP_VALUE)})
    return pd.DataFrame(rows)


def assessment_text(row, shap_rows, layer, reference):
    company_id=str(row['COMPANY_ID']);p=company_probability(layer,company_id)
    lines=[f'# {company_id} — otomatik risk değerlendirmesi',
           f"Risk sırası: {int(row['RISK_RANK'])}/1170. İlk %10 kapasite listesinde: {'Evet' if int(row['ANALYST_REVIEW']) else 'Hayır'}."]
    if p:
        lines.extend([f"Bir yıllık iflas olasılığı tahmini: {percent(p['estimate'])}.",p['status']+'.'])
        band=p['validation_band']
        if band:
            lines.append(f"Aynı tahmin bandında ({percent(band['lower'])}–{percent(band['upper'])}) dış katlardaki {band['n']} kaydın {band['events']} tanesi iflas etiketli: gözlenen oran {percent(band['observed_rate'])}. Bu grup sıklığı bireysel kesinlik değildir.")
    lines.append('\n## Finansal kanıt')
    evidence=evidence_table(shap_rows,reference)
    for _,r in evidence.iterrows():
        observed='eksik' if pd.isna(r['Gözlenen değer']) else f"{r['Gözlenen değer']:.5g}"
        median=r['Geliştirme medyanı'];ref='bilinmiyor' if median is None else f'{median:.5g}'
        lines.append(f"- {r['Değişken']} — {r['Finansal gösterge']}: gözlenen {observed}; model girdisi {r['Modelin kullandığı değer']:.5g}; geliştirme medyanı {ref}; {r['Model etkisi'].lower()} (SHAP {r['SHAP (log-odds)']:+.3f}). {r['Girdi kaynağı']}.")
    lines.append('\n## Sonucun kapsamı\nOlasılık, tarihsel Polonya verisine dayalı keşifsel bir tahmindir. SHAP ham sıralama modelini açıklar; kalibre edilmiş olasılığa yüzde puan katkısı veya nedensel etki göstermez. Gerçek şirkette güncel mali tablo ve veri kapsamı doğrulanmalıdır. Bu rapor gerçekleşmiş iflas etiketini kullanmaz.')
    return '\n\n'.join(lines)


def probability_answer(question, company):
    """Fast path for complete, unambiguous numeric questions only.

    Method questions, negations and compound requests stay in the AI route.
    These are exact supported shortcuts, not a keyword-based intent classifier.
    """
    normalized=unicodedata.normalize('NFKD',str(question).lower().replace('ı','i'))
    normalized=''.join(c for c in normalized if not unicodedata.combining(c))
    codes = re.findall(r'(?<!\w)esik[\s\-–:]*(\d{1,5})(?!\w)', normalized)
    if codes:
        ids = {f'ESIK-{int(code):05d}' for code in codes}
        if len(ids) != 1 or not company or ids != {company.get('company_id')}:
            return None
        normalized = re.sub(r'^esik[\s\-–:]*\d{1,5}\s+(?:icin\s+)?', '', normalized)
    normalized = ' '.join(normalized.strip(' \t\n?.!').split())
    shortcuts = {
        'bir yillik iflas olasiligi tahmini kac', 'bir yillik iflas olasiligi kac',
        'bir yillik iflas ihtimali kac', 'iflas olasiligi kac', 'iflas olasiligi ne',
    }
    if normalized not in shortcuts:return None
    p=company.get('bankruptcy_probability') if company else None
    if not p:return None
    lines=[f"{company['company_id']} için bir yıllık iflas olasılığı tahmini **{percent(p['estimate'])}**.",
        'Modelin ham olasılık tahmini, kat dışı tahminlerden öğrenilen sigmoid ile ayarlanmıştır. '+p['status']+'.']
    b=p['validation_band']
    if b:lines.append(f"Aynı tahmin bandındaki {b['n']} dış doğrulama kaydının {b['events']} tanesi iflas etiketliydi; gözlenen oran {percent(b['observed_rate'])}. Bu, grubun sıklığıdır; tek şirkete ait güven aralığı değildir.")
    if company.get('risk_increasing_factors'):
        lines.append('Ham model skorunu en fazla artıran göstergeler: '+', '.join(f['description']+(' (eksik, medyanla dolduruldu)' if f['was_missing'] else '') for f in company['risk_increasing_factors'][:3])+'.')
    lines.append('Yanıt doğrulanmış hesaplama çıktısından üretildi. SHAP nedensellik göstermez.')
    return '\n\n'.join(lines)


def render_company_assessment(layer,row,shap_rows):
    if layer is None:return
    import streamlit as st
    p=company_probability(layer,str(row['COMPANY_ID']))
    if p is None:return
    st.subheader('Bir yıllık iflas olasılığı ve otomatik değerlendirme')
    a,b=st.columns([1,2])
    a.metric('Kalibrasyonla olasılık tahmini',percent(p['estimate']))
    a.caption('Keşifsel tahmin · tarihsel veri')
    band=p['validation_band']
    if band:
        b.write(f"**Bu yüzdeyi nasıl yorumluyoruz?** {percent(band['lower'])}–{percent(band['upper'])} tahmin bandındaki **{band['n']} dış kat kaydında**, gözlenen iflas oranı **{percent(band['observed_rate'])}**.")
        b.caption(f"Grup oranının yaklaşık %95 Wilson aralığı: {percent(band['wilson95_lower'])}–{percent(band['wilson95_upper'])}. Bu aralık şirketin bireysel olasılığına ait değildir.")
    st.caption(p['status']+'. Ham skor ve risk yüzdeliği bu olasılıktan farklıdır.')
    reference=json.loads((layer['folder']/'development_reference.json').read_text(encoding='utf-8'))
    evidence=evidence_table(shap_rows,reference)
    st.write('**Finansal dayanak:** Model skorunu en fazla artıran üç ve azaltan iki gösterge, geliştirme grubundaki medyanlarıyla birlikte:')
    st.dataframe(evidence,width='stretch',hide_index=True)
    st.caption('SHAP ham modelin log-odds katkısıdır; olasılığa yüzde puan katkısı değildir. Eksik oranlar gözlenmiş değer gibi yorumlanmaz.')
    from esik_report import build_report, render_html
    family=layer['report'].get('family','xgboost')
    model_name={'xgboost':'XGBoost','lightgbm':'LightGBM'}.get(family,family)
    report_html=render_html(build_report(row,shap_rows,layer,model_name))
    st.download_button('Şirketin analiz raporunu indir',report_html,
                       file_name=f"{row['COMPANY_ID']}_inceleme_raporu.html",mime='text/html')
    st.caption('Tasarımlı rapor tarayıcıda açılır. Yazdır menüsünden PDF olarak da kaydedilebilir.')
    with st.expander('Olasılık hesabının yöntemi ve doğrulaması'):
        report=layer['report'];raw=report['outer_raw'];cal=report['outer_calibrated']
        st.write('Sınıf ağırlıklı model zaten ham olasılık tahmini verir. Bu tahminleri, eğitim dışı tahminler üzerinden öğrenilen sigmoid fonksiyonla ayarlıyoruz. Kalibratör sınıf ağırlığı kullanmıyor. Her dış değerlendirme katının etiketleri o katın kalibratör eğitiminden dışlanıyor.')
        st.dataframe(pd.DataFrame([{'Ölçüm':'Ham çıktı','Brier':raw['brier'],'Log loss':raw['log_loss']},{'Ölçüm':'Kalibrasyon sonrası','Brier':cal['brier'],'Log loss':cal['log_loss']}]),hide_index=True,width='stretch')
        st.caption('5 dış kattaki 4.680 tahmin. Brier ve log loss için düşük daha iyi; bunlar yalnız kalibrasyonu ölçmez. İncelenmiş katları yeniden kullanan keşifsel ek deneydir. Yeni bağımsız dış doğrulama değildir.')


def render_calibration_validation(layer):
    if layer is None:return
    import streamlit as st
    import altair as alt
    report=layer['report'];st.subheader('Olasılık kalibrasyonunun doğrulanması')
    st.caption('5 dış katta 4.680 geliştirme kaydı. Önceden incelenmiş katlarla keşifsel ek değerlendirme; tarihsel test bu kalibrasyon eğitimine girmedi.')
    rows=[]
    for key,name in [('raw','Ham çıktı'),('calibrated','Kalibrasyon sonrası')]:
        for b in report['outer_reliability'][key]:rows.append(dict(Seri=name,**b))
    frame=pd.DataFrame(rows)
    diagonal=alt.Chart(pd.DataFrame({'x':[0.,1.],'y':[0.,1.]})).mark_line(color='#999999',strokeDash=[5,5]).encode(x='x:Q',y='y:Q')
    chart=alt.Chart(frame).mark_line(point=True).encode(
        x=alt.X('mean_prediction:Q',title='Ortalama tahmin',scale=alt.Scale(domain=[0,1]),axis=alt.Axis(format='%')),
        y=alt.Y('observed_rate:Q',title='Gözlenen iflas oranı',scale=alt.Scale(domain=[0,1]),axis=alt.Axis(format='%')),
        color=alt.Color('Seri:N',scale=alt.Scale(range=['#FF806C','#BEAEFF'])),
        tooltip=['Seri','n','events',alt.Tooltip('mean_prediction:Q',format='.1%'),alt.Tooltip('observed_rate:Q',format='.1%')])
    st.altair_chart((diagonal+chart).properties(height=340),width='stretch')
    scores=[]
    for key,name in [('outer_raw','Ham çıktı'),('outer_calibrated','Kalibrasyon sonrası')]:scores.append({'Ölçüm':name,'Brier':report[key]['brier'],'Log loss':report[key]['log_loss']})
    st.dataframe(pd.DataFrame(scores),hide_index=True,width='stretch')
    st.caption('Kesikli çizgi ideal eşleşmedir. Noktalar grupların ortalamalarıdır; grup büyüklükleri üzerine gelince görünür. Brier ve log loss hem kalibrasyona hem ayrım gücüne bağlıdır.')
