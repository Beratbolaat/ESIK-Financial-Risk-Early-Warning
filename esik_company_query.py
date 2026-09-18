"""Read-only, model-grounded company lookup shared by the app and message channels."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import unicodedata

import pandas as pd

from esik_artifacts import validate_artifacts
from esik_calibration import load_probability_layer, company_probability
from esik_assessment import probability_answer, percent
from esik_chat_context import build_model_validation
from n8n_client import N8nClientError, send_chat_message

HELP = "Şirket kodunu sorunuza ekleyin. Örnek: ESIK-05511 için en önemli 3 risk nedir? Kodlar tarihsel demo kayıtlarına aittir; gerçek şirket adıyla güncel veri sorgulanmaz."
CODE = re.compile(r"(?<!\w)E[SŞ][Iİ]K[\s\-–:]*(\d{1,5})(?!\w)", re.IGNORECASE)


class CompanyQueryError(ValueError):
    pass


def extract_company_id(question: str) -> str:
    if not isinstance(question, str) or not question.strip():
        raise CompanyQueryError(HELP)
    if len(question) > 2000:
        raise CompanyQueryError("Sorunuzu 2.000 karakterden kısa tutun.")
    normalized = unicodedata.normalize('NFKC', question)
    codes = {f"ESIK-{int(match):05d}" for match in CODE.findall(normalized)}
    if len(codes) > 1:
        raise CompanyQueryError("Her mesajda tek şirket kodu kullanın; böylece hangi kaydın açıklandığı net olur.")
    if not codes:
        raise CompanyQueryError(HELP)
    return codes.pop()


def _number(value):
    return None if pd.isna(value) else float(value)


class CompanyQuery:
    def __init__(self, project_root: Path, *, chat_url: str = '', chat_token: str = ''):
        self.root = Path(project_root)
        manifest = validate_artifacts(self.root)
        self.model_hash = manifest['model_sha256']
        self.probability_layer = load_probability_layer(self.root, self.model_hash)
        out = self.root/'outputs'
        self.rows = pd.read_csv(out/'esik_scored_test_companies.csv').set_index('COMPANY_ID')
        if not self.rows.index.is_unique:
            raise ValueError('Company IDs must be unique')
        self.shap = pd.read_csv(out/'esik_shap_local_values.csv')
        self.metadata = json.loads((out/'esik_model_metadata.json').read_text(encoding='utf-8'))
        self.chat_url = chat_url.strip()
        self.chat_token = chat_token.strip()

    def resolve_company_id(self, question, selected_company_id):
        """Explicit codes win; reject ambiguity before any answer or AI request."""
        if not isinstance(question, str) or not question.strip() or len(question) > 2000:
            raise CompanyQueryError('Sorunuzu 1–2.000 karakter arasında yazın.')
        normalized = unicodedata.normalize('NFKC', question)
        candidates = re.findall(r'(?<!\w)E[SŞ][Iİ]K[\s\-–:]*\d+\w*', normalized, re.IGNORECASE)
        if any(not CODE.fullmatch(candidate) for candidate in candidates):
            raise CompanyQueryError('Şirket kodu geçersiz. Örnek: ESIK-05511.')
        if CODE.search(normalized):
            company_id = extract_company_id(normalized)
        elif re.search(r'(?<!\w)E[SŞ][Iİ]K[\s\-–:]*\d', normalized, re.IGNORECASE):
            raise CompanyQueryError('Şirket kodu geçersiz. Örnek: ESIK-05511.')
        else:
            company_id = selected_company_id
        if company_id not in self.rows.index:
            raise CompanyQueryError(f'{company_id} kayıtlı portföyde bulunamadı; başka şirketin sonucu kullanılmadı.')
        return company_id

    def model_validation(self):
        return build_model_validation(self.root, self.metadata, self.probability_layer)

    def company_context(self, company_id, latest_what_if=None):
        if company_id not in self.rows.index:
            raise CompanyQueryError(f"{company_id} kayıtlı portföyde bulunamadı. Başka şirketin verisiyle yanıt üretilmedi.")
        row = self.rows.loc[company_id]
        values = self.shap[self.shap['COMPANY_ID'] == company_id]
        def factors(sign):
            chosen = values[values['SHAP_VALUE'] * sign > 0].nlargest(5, 'ABS_SHAP_VALUE')
            return [dict(feature=r.VARIABLE, description=r.DESCRIPTION,
                         value=_number(r.MODEL_INPUT_VALUE), shap=float(r.SHAP_VALUE),
                         was_missing=str(r.WAS_MISSING).lower() == 'true')
                    for r in chosen.itertuples()]
        feature_columns = self.metadata['feature_columns']
        return dict(
            company_id=company_id, risk_score=float(row.RISK_SCORE),
            risk_score_scale='0–1 ham iflas olasılığı tahmini; kalibre edilmemiş',
            raw_probability=float(row.RISK_SCORE),
            raw_probability_display=('%'+f'{row.RISK_SCORE*100:.2f}').replace('.', ','),
            risk_score_display=(f'{row.RISK_SCORE*100:.2f}/100').replace('.', ','),
            risk_rank=int(row.RISK_RANK), portfolio_size=len(self.rows),
            risk_group=str(row.RISK_GROUP),
            analyst_review_top_10=bool(int(row.ANALYST_REVIEW)),
            missing_feature_count=int(row[feature_columns].isna().sum()),
            feature_count=len(feature_columns),
            missing_features=row[feature_columns].index[row[feature_columns].isna()].tolist(),
            risk_increasing_factors=factors(1), risk_reducing_factors=factors(-1),
            latest_what_if=latest_what_if if latest_what_if and latest_what_if.get('company_id') == company_id else None,
            bankruptcy_probability=company_probability(self.probability_layer, company_id),
        )

    def _summary(self, company):
        yes_no = 'Evet' if company['analyst_review_top_10'] else 'Hayır'
        lines = [
            f"{company['company_id']} | sıra {company['risk_rank']}/{company['portfolio_size']}",
            f"Model skoru (0–100): {company['risk_score']*100:.4g}",
            f"İlk %10 inceleme listesinde: {yes_no}",
            f"Eksik girdi: {company['missing_feature_count']}/64",
            'Skoru yukarı çeken başlıca göstergeler:',
        ]
        if company.get('bankruptcy_probability'):
            p = company['bankruptcy_probability']
            lines.insert(2, f"Bir yıllık iflas olasılığı tahmini: {percent(p['estimate'])}. Keşifsel kalibrasyon; tarihsel veri.")
        for i, factor in enumerate(company['risk_increasing_factors'][:3], 1):
            value = 'eksik; eğitim medyanıyla doldurulmuş' if factor['was_missing'] else f"{factor['value']:.4g}" if factor['value'] is not None else 'eksik'
            lines.append(f"{i}. {factor['feature']} — {factor['description']} ({value})")
        if not company['risk_increasing_factors']:
            lines.append('Bu kayıtta pozitif SHAP katkısı bulunmuyor.')
        lines.append('Analist, bu göstergelerin dayandığı finansal kayıtları ve eksik girdileri kontrol edebilir.')
        return '\n'.join(lines)

    def query(self, question: str, *, session_id: str = 'single-question'):
        try:
            company_id = extract_company_id(question)
            company = self.company_context(company_id)
        except CompanyQueryError as error:
            return {'status':'needs_input', 'answer':str(error), 'answer_mode':'validation', 'company_id':None}
        metadata = self.metadata
        validation = self.model_validation()
        payload = dict(schema_version='1.0', session_id=session_id,
                       question=question, context={'context_type':'company', 'company':company, 'portfolio':None},
                       conversation_history=[], model_validation=validation)
        answer = self._summary(company)
        mode = 'record_summary'
        calculated_answer = probability_answer(question, company)
        if calculated_answer:
            answer = calculated_answer
            mode = 'calibrated_probability_report'
        elif self.chat_url:
            try:
                interpretation = send_chat_message(webhook_url=self.chat_url, payload=payload,
                                                   webhook_token=self.chat_token, timeout_seconds=30)
                answer += '\n\nAsistan yorumu:\n' + interpretation
                mode = 'model_grounded_ai'
            except N8nClientError:
                answer += '\n\nAI yorum hizmetine ulaşılamadı; yukarıda kayıtlı model özeti gösteriliyor.'
                mode = 'record_summary_fallback'
        else:
            answer += '\n\nBu yanıt kayıtlı model özetidir; bu sorguda AI yorumu kullanılmadı.'
        source = dict(dataset=metadata['dataset'], model_version=metadata['model_version'],
                      model_sha256=self.model_hash, data_scope='Tarihsel ve anonim Polonya kayıtları; güncel şirket verisi değil',
                      queried_at_utc=datetime.now(timezone.utc).isoformat())
        answer += f"\n\nKaynak: {source['dataset']}. Model: {source['model_version']}. Tarihsel demo kaydı. Ham skor, kalibre edilmemiş model olasılık tahminidir; SHAP nedensellik göstermez."
        return dict(status='ok', answer=answer[:3900], answer_mode=mode, company_id=company_id,
                    context=payload['context'], source=source)


def private_session_id(channel: str, sender: str) -> str:
    """Do not pass a phone number/chat ID to the language model."""
    return hashlib.sha256(f'{channel}:{sender}'.encode()).hexdigest()
