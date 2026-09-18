from pathlib import Path
import copy
from esik_company_query import CompanyQuery
from esik_report import build_report,render_html,render_text

def test_report_escapes_text_and_preserves_missing_evidence():
    q=CompanyQuery(Path(__file__).resolve().parents[1])
    row=q.rows.loc['ESIK-05817'].copy();row['COMPANY_ID']='ESIK-05817'
    d=build_report(row,q.shap[q.shap.COMPANY_ID==row.COMPANY_ID],q.probability_layer,'XGBoost')
    assert d['missing']==2 and d['rank']==3
    assert any(f['feature']=='Attr21' and f['observed'] is None for f in d['factors'])
    html=render_html(d);text=render_text(d)
    assert 'Eğitim medyanı kullanıldı' in html and '%99,0' in html
    assert '##' not in text and '**' not in text
    changed=copy.deepcopy(d);changed['model']='<script>alert(1)</script>'
    assert '<script>' not in render_html(changed) and '&lt;script&gt;' in render_html(changed)
    assert all(token not in html.lower() for token in ['<script','<iframe','javascript:','src=','href='])
