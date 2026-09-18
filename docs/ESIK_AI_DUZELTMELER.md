# EŞİK AI düzeltmeleri — 15 Eylül 2026

XGBoost sürümünün sohbet yönlendirmesi, veri aktarımı ve n8n açıklama talimatları düzeltildi. Çalışan v3 n8n akışı “EŞİK sohbet doğruluk düzeltmesi” adıyla yayımlandı; taşınabilir v1/v3 iş akışı dosyaları da güncellendi.

## Düzeltilen davranışlar

- **Sorudaki şirket kodu öncelikli.** Başka şirket sorulduğunda o kayıt bulunuyor, ekrandaki seçim değişiyor ve cevap aynı kayıttan üretiliyor. Geçersiz, bulunamayan veya birden fazla şirket kodunda yanlış kayda düşülmüyor. Şirket/model değiştiğinde eski sohbet bağlamı temizleniyor.
- **Ham olasılık doğru tanımlanıyor.** `predict_proba` zaten ham, kalibre edilmemiş olasılık tahminidir. Sigmoid kalibrasyon mevcut tahmini ayarlar. Bu ayrım veri alanlarında, n8n isteminde ve uygulama açıklamalarında açık.
- **Geniş kelime eşleştirmesi kaldırıldı.** “Kalibrasyonu neden yaptık?” ve “Olasılığı tekrar yazma, SHAP'ı açıkla” gibi sorular hazır şirket yüzdesine takılmıyor. Yalnız tam ve tek anlamlı sayısal sorular yerel hesaplamadan yanıtlanıyor; diğerleri mevcut AI bağlantısına gidiyor.
- **İki giriş aynı bilgiyi kullanıyor.** Normal sohbet ve kodla sorgu; model karşılaştırması, seçim kuralı, kalibrasyon yöntemi/ölçümleri, uygulama bölümleri, 64 girdinin toplam eksik sayısı ve eksik değişken listesini aynı kaynaktan alıyor.
- **What-if şirketle eşleşiyor.** Başka şirketin senaryosu aktarılmıyor. Senaryo yoksa asistan sonucu uydurmak yerine nerede oluşturulacağını açıklıyor.
- **Konuşma dili düzeltildi.** Kısa selamlaşmalara doğal cevap verilmesi, uygulama içi alan adlarının ve gereksiz uyarı tekrarlarının azaltılması için talimatlar yenilendi.

## Doğrulama

91 otomatik test geçti. Streamlit'in sekiz sayfası hatasız açıldı. Önceki incelemedeki 12 soru yeniden denendi: 10 canlı n8n/OpenAI yanıtı ve iki yerel olasılık yanıtı incelendi. Ham/kalibre ayrımı, yöntem açıklaması, XGBoost–LightGBM rakamları, eksik sayısı ve şirket eşleştirmesi doğrulandı.

Gerçek 8503 ekranında ESIK-05817 seçiliyken “ESIK-05511 için bir yıllık iflas ihtimali kaç?” yazıldı. Seçim ESIK-05511'e geçti; sıra 125, ham skor 17.4/100 ve kalibre tahmin %14,7 olarak görüldü. Cevapta da aynı şirket ve %14,7 yer aldı. Bu belirli soru yerel sayısal yanıt yolunda işlendi.

Model dosyasının SHA-256 özeti değişmedi. Bu bir yeniden eğitim veya performans artışı iddiası değildir. Önceki denetimde 1.170 şirketin skor/kalibre tahminleri ve 74.880 SHAP katkısı modelden yeniden hesaplanarak kontrol edilmişti. Tarihsel XGBoost sonucu hâlâ 82 iflasın 67'sini 117 kayıtlık listede yakalamadır.

Kanıtlar `docs/ai_duzeltme/` klasöründe; otomatik test kaydı `docs/chat-fix-tests.xml` dosyasında. Üretken cevapların üslubu ve tablo vurguları değişebilir; bu kontroller tüm sorulara hatasız cevap garantisi değildir. Yeni ses kaydı bu turda denenmedi.

## Kullanım

Bu bilgisayarda çalışan güncel 8503 sayfasını yenileyin. Eski sohbet açık kaldıysa “Sohbeti temizle” düğmesiyle yeni konuşma başlatın. Mevcut n8n bağlantısı güncellendi; anahtarı yeniden girmeniz gerekmez.

Paylaşılacak ZIP güncel kodları, modelleri, sunumu ve n8n şablonlarını içerir. Kişisel `secrets.toml` ve API anahtarları paylaşım paketine eklenmez. Başka bilgisayarda mevcut kurulum belgeleri izlenmeli ve bağlantı o bilgisayarda yapılandırılmalıdır.
