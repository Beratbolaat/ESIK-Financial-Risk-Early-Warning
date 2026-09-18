# EŞİK · 16 Eylül güncellemesi

PowerPoint: ESIK_XGBoost_Sahne_Final_16_Eylul_2026.pptx. Konuşmacı notları sunumun içinde ve docs/sunum/KONUSMACI_NOTLARI.md dosyasında.

## Uygulama ve rapor

Analist Öncelik Listesi sayfasında bir satır seçin; beliren düğme ile o şirketin detayına geçin. Şirket sayfasındaki Finansal kanıt, Senaryo analizi ve Rapor paylaşımı bağlantıları ilgili bölüme götürür. Hareketi azaltma işletim sistemi tercihi desteklenir.

Şirket Detayı > Analiz raporunu e-postayla gönder bölümünde raporun gerçek HTML önizlemesi ve temiz metin alternatifi vardır. Tasarımlı raporu indir düğmesi, bağımsız açılabilen bir HTML dosyası verir. Tarayıcı Yazdır > PDF olarak kaydet ile paylaşılabilir. Gönderim yalnız E-postayla gönder düğmesiyle başlar. SMTP kabulünden sonra tekrar gönderim engellenir. Sekme/sayfa yenilemek otomatik e-posta göndermez.

Canlı n8n’de mevcut EŞİK - Şirket Raporunu E-postala akışı güncellendi. Yeni/ikinci akış açılmadı. Gmail ile Rapor Gönder düğümü Both biçiminde; HTML alanı report_html, metin alanı report_text kullanır. Rapor ve Alıcı Kontrolü düğümü HTML sürümünü ve içerik sınırlarını doğrular. Mevcut SMTP bağlantısı, alıcı ve webhook kimlik doğrulaması korunur.

Başka bilgisayarda n8n/esik-email-workflow.json taşınabilir şablondur: kendi gönderen/alıcı adresinizi ve mevcut SMTP/Header Auth bağlantılarınızı seçip yayımlayın. Anahtarlar ZIP’e eklenmemiştir. Mevcut secrets.toml dosyanızı koruyun.

## Küçük güncelleme

Uygulamayı kapatıp küçük güncellemedeki esik-profesyonel klasörü içeriğini mevcut XGBoost proje klasörünüzün üzerine kopyalayın. app.py, esik_email.py, esik_report.py, esik_assessment.py, esik_ui.py, assets/esik.css ve n8n/esik-email-workflow.json birlikte güncellenmelidir. Model yeniden eğitilmez. Bu bilgisayardaki aktif proje zaten güncellendi.

Bu bilgisayarda uygulama http://127.0.0.1:8503/ ve n8n http://127.0.0.1:5678/ adresinde çalışır. Tekrar açmak için kendi XGBoost proje klasörünüzde python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8503 komutunu kullanın; n8n için ayrı terminalde n8n.cmd start.

## Doğrulama

Sekiz Streamlit sayfası AppTest ile açıldı. Dokuz rapor/e-posta testi geçti; raporda gerçekleşmiş şirket etiketi kullanılmaması, eksik girdilerin korunması, HTML kaçışı ve SMTP kabul denetimi kontrol edildi. 390 piksel genişlikte raporda yatay taşma yok. Canlı n8n çalışma #79, 16 Eylül 2026 11:21:31: başarılı, 2,441 saniye. Önizlenen ESIK-05817 raporu kullanıcının kendi Gmail adresine bir kez gönderildi; SMTP kabulü görüldü. Gelen kutusu görünümü ayrıca okunmadı.

Sunumda 19 slayt, sekiz düzenlenebilir grafik ve iki düzenlenebilir tablo korunur. Kaynak grafik değerleri ve gömülü çalışma kitapları değişmedi. Son slaytlar tek tek görsel olarak kontrol edildi. Animasyon tanımları ve hedef nesne kimlikleri doğrulandı; masaüstü PowerPoint’te hareket oynatımı bu ortamda test edilmedi. Mevcut ses/AI ve model eğitimi yeniden çalıştırılmadı.
