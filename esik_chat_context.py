"""Shared factual context for both Eşik chat entry points; no test labels per company."""
from pathlib import Path
import json

CHAT_CONTEXT_VERSION = 'esik-chat-v2'

APPLICATION_GUIDE = {
    'Şirket Detayı': 'Şirket skoru, kalibre olasılık, SHAP, eksik girdiler ve What-if senaryosu.',
    'Analist Öncelik Listesi': 'Risk sırasına göre şirketler ve ilk %10 inceleme listesi.',
    'Model Doğrulaması': 'Tarihsel test metrikleri ve olasılık kalibrasyonu değerlendirmesi.',
    'Model Kartı': 'Model ailesi, veri kapsamı, seçim ve doğrulama sınırlamaları.',
    'Yeni Veri Analizi': '64 finansal oranı içeren CSV yükleme ve ham model skorları.',
    'Eşik AI': 'Seçili şirket veya mesajdaki şirket kodu için mevcut sonuçların açıklaması.',
}


def build_model_validation(root, metadata, probability_layer):
    """Send compact verified summaries, never raw train/test label rows."""
    root = Path(root)
    result = {key: metadata[key] for key in (
        'model_name', 'model_version', 'validation_status', 'feature_dictionary_note',
        'test_roc_auc', 'test_average_precision', 'test_recall_at_top_10',
        'test_precision_at_top_10', 'test_lift_at_top_10') if key in metadata}
    result.update(
        context_version=CHAT_CONTEXT_VERSION,
        score_is_calibrated_probability=False,
        score_definition='risk_score = predict_proba(X)[:, 1]: ham, kalibre edilmemiş iflas olasılığı tahmini. 100 ile çarpımı yalnız gösterim ölçeğidir.',
        calibrated_probability_definition='bankruptcy_probability.estimate: ham marja uygulanan, geliştirme verisindeki kat dışı tahminlerden öğrenilmiş sigmoid sonucu.',
        dataset=metadata.get('dataset'),
        prediction_target=metadata.get('prediction_target'),
        feature_count=metadata.get('feature_count'),
        development_rows=len(metadata.get('development_ids', [])),
        historical_test_rows=len(metadata.get('holdout_ids', [])),
        best_parameters=metadata.get('best_parameters', {}),
        application_guide=APPLICATION_GUIDE,
        data_scope='Tarihsel, anonim Polonya şirket kayıtları. Canlı haber veya güncel şirket verisi sorgulanmıyor.',
        metric_definitions={
            'recall_at_top_10': 'İlk %10 listede yakalanan gerçek iflas / toplam gerçek iflas. Genel doğruluk veya tek şirket olasılığı değildir.',
            'precision_at_top_10': 'İlk %10 listede yakalanan gerçek iflas / listedeki şirket sayısı.',
            'average_precision': 'Farklı eşiklerdeki kesinliğin yakalamadaki artışla ağırlıklandırılmış özeti. Sabit bir listedeki isabet değildir.',
            'brier_and_log_loss': 'Olasılık tahmin hataları; düşük daha iyi. Yalnız kalibrasyonu ölçmezler.',
            'shap': 'Ham model marjına log-odds katkısı. Kalibre olasılığa yüzde puan katkısı veya nedensellik değildir.',
        },
        preprocessing='Eksik girdiler eğitim bölümünde öğrenilen medyanla doldurulur; doğrulama/testte yeniden medyan öğrenilmez.',
    )
    path = root/'karsilastirma/comparison_report.json'
    if path.exists():
        comparison = json.loads(path.read_text(encoding='utf-8'))
        selected = comparison['models'].get(metadata.get('validation_key'), {})
        selected_hash = selected.get('calibration', {}).get('model_sha256')
        if probability_layer and selected_hash != probability_layer['report'].get('model_sha256'):
            raise ValueError('Sohbet karşılaştırması ile çalışan model sürümü uyuşmuyor.')
        result['model_comparison'] = {
            family: {
                'outer_cv': {k: v for k, v in values['outer'].items() if k != 'seconds'},
                'historical_test': values['historical_ranking'],
                'final_inner_recall_at_10': values['final_inner']['mean_recall_at_10'],
            } for family, values in comparison['models'].items()
        }
        protocol = comparison['protocol']
        result['selection'] = {
            'chosen_model': metadata['model_name'],
            'criterion': 'Bu karşılaştırma sürümünde ailelerin beş dış kat ortalama Recall@Top10 değeri; yüksek daha iyi.',
            'status': 'Aile tercihi daha önce incelenmiş dış katlar üzerinden sonradan yapıldı. Başlangıçtan beri aynı seçim kuralını kullandık denmemeli. Eski ortak iç arama LightGBM seçmişti.',
            'limitation': 'Bu katlar yeni bağımsız doğrulama değildir; tarihsel test de daha önce görüldü. Tarihsel testte LightGBM önde. Kesin genel üstünlük gösterilmedi.',
            'outer_folds': protocol['outer_folds'],
            'inner_folds': protocol['calibration_inner_folds'],
            'original_optuna_trials_per_family': protocol['original_trials_per_family_per_search'],
            'new_optuna_trials_in_comparison': protocol['new_hyperparameter_trials'],
        }
    if probability_layer:
        report = probability_layer['report']
        result['calibration'] = {
            'method': 'Pozitif eğimli sigmoid: ham model marjı ile eğitim dışı tahminlerin gerçek 0/1 etiketleri arasındaki ilişki öğrenilir; sınıf ağırlığı kullanılmaz.',
            'need': 'Model zaten ham olasılık verir. Sınıf ağırlıkları ve model uyumu yüzdeleri etkileyebilir; sıralama başarısı doğru olasılık yüzdelerini garanti etmez.',
            'calibration_rows': len(metadata.get('development_ids', [])),
            'raw': report['outer_raw'], 'calibrated': report['outer_calibrated'],
            'ranking_preserved': report['original_ranking_preserved'],
            'scope': 'İncelenmiş geliştirme katlarında keşifsel ek deney. Yeni bağımsız test değildir. Tarihsel test kalibratör eğitiminde kullanılmadı.',
            'band_note': 'Gözlenen iflas sıklığı ve güven aralığı aynı tahmin bandındaki gruba aittir; tek şirketin güven aralığı değildir.',
        }
    return result
