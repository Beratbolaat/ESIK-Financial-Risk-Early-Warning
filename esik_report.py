"""One structured report for HTML email, browser preview and a clean text alternative."""
from datetime import date
from html import escape
import hashlib
import json
import re

from esik_assessment import evidence_table, percent
from esik_calibration import company_probability

REPORT_VERSION = 'esik-report-2'


def number(value):
    return 'Eksik' if value is None else f'{float(value):.4f}'.replace('.', ',')


def build_report(row, shap_rows, layer, model_name):
    company_id = str(row['COMPANY_ID'])
    if not re.fullmatch(r'ESIK-\d{5}', company_id):
        raise ValueError('Geçersiz şirket kodu.')
    probability = company_probability(layer, company_id)
    if probability is None:
        raise ValueError('Doğrulanmış olasılık kaydı bulunamadı.')
    reference = json.loads((layer['folder']/'development_reference.json').read_text(encoding='utf-8'))
    table = evidence_table(shap_rows, reference)
    import pandas as pd
    factors = []
    for _, item in table.iterrows():
        factors.append(dict(feature=str(item['Değişken']), label=str(item['Finansal gösterge']),
            observed=None if pd.isna(item['Gözlenen değer']) else float(item['Gözlenen değer']),
            model_input=float(item['Modelin kullandığı değer']), median=item['Geliştirme medyanı'],
            shap=float(item['SHAP (log-odds)']), effect=str(item['Model etkisi'])))
    missing = int(shap_rows['WAS_MISSING'].astype(str).str.lower().eq('true').sum())
    review = bool(int(row['ANALYST_REVIEW']))
    data = dict(version=REPORT_VERSION, company_id=company_id, model=str(model_name),
        date=date.today().strftime('%d.%m.%Y'), portfolio_size=len(layer['rows']),
        rank=int(row['RISK_RANK']), group=str(row['RISK_GROUP']), review=review,
        raw_probability=float(row['RISK_SCORE']), probability=probability['estimate'],
        missing=missing, feature_count=len(shap_rows), factors=factors,
        band=probability['validation_band'], status=probability['status'])
    # A reproducible identifier without including a recipient or historical outcome.
    data['report_id'] = hashlib.sha256(json.dumps(data,ensure_ascii=False,sort_keys=True).encode()).hexdigest()[:10].upper()
    return data


def render_text(data):
    d=data
    lines=['EŞİK — FİNANSAL İNCELEME RAPORU', d['company_id'],
        f"Rapor tarihi: {d['date']} | Model: {d['model']} | Rapor no: {d['report_id']}", '',
        'İNCELEME ÖZETİ',
        f"Risk sırası: {d['rank']}/{d['portfolio_size']}. Risk grubu: {d['group']}.",
        'İlk %10 inceleme listesinde: '+('Evet.' if d['review'] else 'Hayır.'),
        f"Kalibre edilmiş bir yıllık olasılık tahmini: {percent(d['probability'])}.",
        f"Ham model olasılığı: {percent(d['raw_probability'])}.",
        f"Eksik model girdisi: {d['missing']}/{d['feature_count']}.", '', 'FİNANSAL DAYANAKLAR']
    for f in d['factors']:
        lines.extend([f"{f['feature']} — {f['label']}",
            f"Gözlenen: {number(f['observed'])}; model girdisi: {number(f['model_input'])}; geliştirme medyanı: {number(f['median'])}.",
            f"{f['effect']}. SHAP: {f['shap']:+.3f} (log-odds)."+(' Eksik değer eğitim medyanıyla dolduruldu.' if f['observed'] is None else ''),''])
    lines.extend(['ANALİST İÇİN İLK KONTROL',
        'Eksik göstergelerin kaynak mali tablolarını doğrulayın.' if d['missing'] else 'Öne çıkan oranları kaynak mali tablolarıyla doğrulayın.',
        'Şirketin dönemini, muhasebe kapsamını ve oranların birlikte tutarlılığını kontrol edin.', '', 'TAHMİNİN KAPSAMI',
        d['status']+'.',
        'Tarihsel ve anonim Polonya verisine dayalı araştırma prototipi. Gerçek şirkette güncel dış doğrulama gerekir.',
        'SHAP ham modelin tahminini açıklar. Olasılığa yüzde puan katkısı veya nedensel etki değildir.',
        'Raporda bu şirketin gerçekleşmiş iflas etiketi kullanılmaz. İnceleme sırası bir karar desteğidir.'])
    return '\n'.join(lines)


def render_html(data):
    d=data; h=lambda x:escape(str(x),quote=True)
    rank=f"{d['rank']} <span style='font-size:16px;color:#727985'>/ {d['portfolio_size']}</span>"
    review='İlk %10 inceleme listesinde' if d['review'] else 'İlk %10 inceleme listesinin dışında'
    body_rows=[]
    for f in d['factors']:
        up=f['shap']>0; color='#aa3c2c' if up else '#276951'
        observed=("<strong style='color:#aa3c2c'>Eksik</strong><br><span style='font-size:11px;color:#757c86'>Eğitim medyanı kullanıldı</span>"
                  if f['observed'] is None else h(number(f['observed'])))
        body_rows.append(f'''<tr>
          <td style="padding:15px 12px;border-bottom:1px solid #e4e7eb;vertical-align:top;width:43%"><strong>{h(f['label'])}</strong><br><span style="color:#757c86;font-size:11px">{h(f['feature'])} · Geliştirme medyanı {h(number(f['median']))}</span></td>
          <td style="padding:15px 8px;border-bottom:1px solid #e4e7eb;vertical-align:top">{observed}</td>
          <td style="padding:15px 8px;border-bottom:1px solid #e4e7eb;vertical-align:top">{h(number(f['model_input']))}</td>
          <td style="padding:15px 10px;border-bottom:1px solid #e4e7eb;vertical-align:top;text-align:right;color:{color}"><strong>{h(f"{f['shap']:+.3f}".replace('.',','))}</strong><br><span style="font-size:11px">{'Artırıyor' if up else 'Azaltıyor'}</span></td>
        </tr>''')
    band=d['band']
    band_text=''
    if band:
        band_text=(f"Aynı tahmin bandındaki {band['n']} dış kat kaydının {band['events']} tanesi iflas etiketliydi. "
                   f"Gözlenen grup oranı {percent(band['observed_rate'])}. Bu grup oranı, şirketin bireysel kesinliği değildir.")
    first_check='Eksik finansal girdileri tamamlayın' if d['missing'] else 'Kaynak mali tabloları doğrulayın'
    return f'''<!doctype html>
<html lang="tr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="esik-report-version" content="{REPORT_VERSION}"><title>EŞİK · {h(d['company_id'])}</title>
<style>body{{margin:0;background:#edf0f3}}table{{border-collapse:collapse}}.report-wrap{{width:100%;max-width:760px}}td{{overflow-wrap:anywhere}}@media(max-width:600px){{.outer-pad{{padding:12px!important}}.section-pad{{padding:22px 18px!important}}.metric-value{{font-size:25px!important}}.evidence{{font-size:11px!important}}}}@media print{{body{{background:white}}.outer-pad{{padding:0!important}}.report-wrap{{max-width:none!important}}tr{{break-inside:avoid}}h2{{break-after:avoid}}.report-footer{{break-inside:avoid}}}}</style></head>
<body style="margin:0;background:#edf0f3;font-family:Arial,'Segoe UI',sans-serif;color:#1b2330;line-height:1.55">
<div style="display:none;max-height:0;overflow:hidden">{h(d['company_id'])} için inceleme sırası, olasılık tahmini ve finansal dayanaklar.</div>
<table role="presentation" width="100%"><tr><td class="outer-pad" align="center" style="padding:30px 20px">
<table role="presentation" class="report-wrap" width="760" style="width:100%;max-width:760px;background:#ffffff;border:1px solid #dce1e7">
<tr><td class="section-pad" style="padding:30px 36px;background:#10141f;border-top:5px solid #d7ff42;color:#f4f5f0">
<table role="presentation" width="100%"><tr><td style="font-size:28px;font-weight:800;letter-spacing:-1px;color:#d7ff42">EŞİK<span style="color:#b6a0ff">.</span></td><td align="right" style="font-size:11px;color:#c3c8d4">FİNANSAL İNCELEME RAPORU<br>{h(d['date'])}</td></tr></table>
<h1 style="font-size:34px;line-height:1.2;margin:27px 0 9px;letter-spacing:-1px">{h(d['company_id'])}</h1>
<p style="margin:0;color:#c3c8d4;font-size:14px">{h(review)}<br>{h(d['model'])} modeli · 1 yıllık tahmin ufku</p></td></tr>
<tr><td class="section-pad" style="padding:28px 36px 12px">
<table role="presentation" width="100%" style="background:#f5f4fb;border-left:3px solid #8063e8"><tr>
<td width="34%" style="padding:18px 12px;vertical-align:top"><span style="font-size:11px;color:#626b7a">İNCELEME SIRASI</span><br><strong class="metric-value" style="font-size:32px;line-height:1.5">{rank}</strong></td>
<td width="36%" style="padding:18px 12px;vertical-align:top"><span style="font-size:11px;color:#626b7a">KALİBRE TAHMİN</span><br><strong class="metric-value" style="font-size:32px;line-height:1.5;color:#5940bb">{h(percent(d['probability']))}</strong><br><span style="font-size:11px;color:#626b7a">Keşifsel olasılık</span></td>
<td width="30%" style="padding:18px 12px;vertical-align:top"><span style="font-size:11px;color:#626b7a">EKSİK GİRDİ</span><br><strong class="metric-value" style="font-size:32px;line-height:1.5">{d['missing']} <span style="font-size:16px;color:#727985">/ {d['feature_count']}</span></strong></td>
</tr></table>
<p style="font-size:12px;color:#626b7a;margin:13px 0 0">Ham model olasılığı {h(percent(d['raw_probability']))}. İnceleme sırası ham model skoruna dayanır. Kalibre tahmin ayrı gösterilir.</p></td></tr>
<tr><td class="section-pad" style="padding:22px 36px 12px">
<h2 style="font-size:20px;margin:0 0 8px;letter-spacing:-.4px">İnceleme özeti</h2>
<p style="font-size:14px;margin:0">Bu kayıt {d['portfolio_size']} şirket içinde {d['rank']}. sırada yer alıyor. {h(d['group'])} grubunda. {'İlk %10 kapasite varsayımı altında analist inceleme listesine giriyor.' if d['review'] else 'İlk %10 kapasite varsayımı altında liste dışında kalıyor. Bu durum iflas olasılığını dışlamaz.'}</p>
<p style="font-size:14px;margin:10px 0 0">{d['missing']} finansal girdi eksik. Eksik girdiler modelde eğitim medyanlarıyla dolduruluyor.</p></td></tr>
<tr><td class="section-pad" style="padding:22px 36px 12px">
<h2 style="font-size:20px;margin:0 0 5px;letter-spacing:-.4px">Tahmini etkileyen göstergeler</h2>
<p style="font-size:12px;color:#626b7a;margin:0 0 14px">Ham skoru en fazla artıran üç ve azaltan iki katkı</p>
<table class="evidence" width="100%" style="font-size:12px;text-align:left;table-layout:fixed"><thead><tr style="background:#eef0f4;color:#525b6a">
<th scope="col" width="43%" style="padding:10px 12px">Finansal gösterge</th><th scope="col" width="20%" style="padding:10px 8px">Gözlenen</th><th scope="col" width="19%" style="padding:10px 8px">Model girdisi</th><th scope="col" width="18%" style="padding:10px 10px;text-align:right">SHAP</th></tr></thead><tbody>{''.join(body_rows)}</tbody></table>
<p style="font-size:11px;color:#626b7a;margin:12px 0 0">SHAP değerleri ham modelin log-odds ölçeğindedir. Olasılığa yüzde puan katkısı veya neden-sonuç ilişkisi göstermez. Eksik girdi gözlenmiş değer gibi yorumlanmaz.</p></td></tr>
<tr><td class="section-pad" style="padding:22px 36px">
<table role="presentation" width="100%" style="background:#f5f7ef;border-left:3px solid #8caa24"><tr><td style="padding:18px 20px"><h2 style="font-size:18px;margin:0 0 8px">Analistin ilk kontrolü</h2><p style="font-size:14px;margin:0"><strong>{h(first_check)}.</strong> Öne çıkan oranların dönemini ve muhasebe kapsamını kontrol edin. Finansal göstergeleri birlikte değerlendirin.</p></td></tr></table>
<p style="font-size:12px;color:#626b7a;margin:18px 0 0">{h(band_text)}</p></td></tr>
<tr><td class="section-pad report-footer" style="padding:22px 36px;background:#f5f6f8;border-top:1px solid #e1e5eb">
<strong style="font-size:12px">Raporun kapsamı</strong><p style="font-size:11px;color:#606a79;margin:7px 0">Tarihsel ve anonim Polonya verisine dayalı araştırma prototipi. {h(d['status'])}. Raporda şirketin gerçekleşmiş iflas etiketi kullanılmaz. Gerçek şirkette güncel mali tablolar ve bağımsız doğrulama gerekir.</p>
<p style="font-size:10px;color:#7b8390;margin:16px 0 0">EŞİK · Ayşe Simal Alpözer ve Berat Bolat<br>Rapor no {h(d['report_id'])} · {REPORT_VERSION}</p></td></tr>
</table></td></tr></table></body></html>'''
