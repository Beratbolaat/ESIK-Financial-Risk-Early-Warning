# EŞİK AI — yerel n8n/OpenAI bağlantısı

Mevcut JSON **OpenAI Chat Model** düğümü ve maliyet odaklı `gpt-5.6-luna` modelini kullanır. Python model skorunu, SHAP'ı ve What-if sonucunu hesaplar; n8n bunları bir dil modeline açıklatır. LLM, iflas modelini eğitmez ve skoru yeniden hesaplamaz.

Bu çalışma sırasında canlı n8n/OpenAI isteği gönderilmedi. Çevrimdışı istemci/JSON testleri geçmesi, canlı entegrasyonun tamamlandığı anlamına gelmez. Ana dashboard bu bağlantı olmadan çalışır.

## Yerel kurulum

1. Mevcut n8n kurulumunuzu açın. Yeni bir workflow'a `n8n/esik-ai-workflow.json` dosyasını içe aktarın. Dosya pasiftir ve kişisel credential referansı içermez; mevcut canlı akış değiştirilmez.
2. **OpenAI Chat Model** düğümünde yeni bir OpenAI credential oluşturun ve OpenAI API anahtarınızı yalnız n8n credential alanına girin. Organization ID tek organizasyon kullanıyorsanız boş kalabilir.
3. Model listesinde hesabınızda görünüyorsa `gpt-5.6-luna` seçin. Görünmüyorsa OpenAI düğümünün hesabınız için yüklediği erişilebilir, düşük maliyetli bir metin modelini seçin ve canlı testten sonra workflow'u yeniden dışa aktarın.
4. **Webhook** düğümü `POST` ve `esik-ai-dev` yolunu kullanır. **Listen for test event** ile düğümün gösterdiği Test URL'sini alın. Yerel örnek: `http://localhost:5678/webhook-test/esik-ai-dev`.
5. `.streamlit/secrets.example.toml` dosyasını `.streamlit/secrets.toml` olarak kopyalayın. `N8N_WEBHOOK_URL` alanına düğümde gördüğünüz adresi yazın. API anahtarı bu dosyaya değil, n8n credential deposuna girilir.
6. Streamlit'i yeniden başlatın. Eşik AI sayfasında şirket seçip kısa bir soru gönderin. Test URL'si dinleyici açıkken çalışır; yeniden dinleme gerekebilir.
7. Yerel sürekli demoda workflow'u kaydedip etkinleştirdikten sonra düğümdeki Production URL'sini kullanın. Yerel örnek: `http://localhost:5678/webhook/esik-ai-dev`. Gerçek adres n8n kurulumunuzun gösterdiği adrestir.

İnternetten erişilebilir webhook gerekiyorsa Webhook düğümünde Header Auth yapılandırın: başlık `X-ESIK-WEBHOOK-TOKEN`. Aynı değeri yerel `N8N_WEBHOOK_TOKEN` ayarında tutun. Secrets dosyası Git'e/ZIP'e eklenmez. Bu paket bir internet yayını yapmaz.

## Kabul kontrolü

| Deneme | Beklenen davranış |
|---|---|
| “Bu şirket neden riskli görünüyor?” | Gönderilen şirketin skorunu ve SHAP faktörlerini açıklamalı |
| Eksik bir girdinin faktör olması | `was_missing=true` ise medyanla doldurulduğunu belirtmeli |
| “Bu şirket kesin iflas eder mi?” | Kalibrasyon ve doğrulama sınırını açıklamalı; kesin hüküm üretmemeli |
| What-if hesaplamadan “Yeni skor ne?” | Sayı uydurmamalı; hesaplanmış senaryo istemeli |
| Hesaplanmış What-if'i yorumlama | Gönderilen skorları korumalı; nedensel sonuç iddia etmemeli |
| Portföy bağlamı | Şirket ve sıra bilgilerini yalnız gönderilen listeden almalı |
| Bağlantı zaman aşımı | Anlaşılır hata göstermeli; dashboard analizi kullanılabilmeli |

Bu kontroller yalnız prompt okuyarak kanıtlanamaz. Sunumdan önce canlı hesapla denenip yanıtları gözden geçirilmelidir. Kimlik bilgileri/API hesabı bu pakette bulunmadığı için bu kabul kontrolü açıktır.

## Sık hatalar

- `404 webhook is not registered`: Test dinleyicisini yeniden açın veya etkin workflow'un Production URL'sini kullanın.
- Bağlantı hatası: n8n çalışıyor mu, URL ve port doğru mu kontrol edin.
- Credential/model erişim hatası: OpenAI düğümünde API anahtarını ve hesabınızın eriştiği modeli seçin; ChatGPT aboneliği API bakiyesi yerine geçmez.
- Yanıt yok: **Yanıt Biçimlendir → Respond to Webhook** bağlantısını ve n8n execution kaydını inceleyin.

LLM açıklamalarının finansal doğruluğu garanti edilmez; son değerlendirme analiste aittir. API kullanımı sağlayıcının kendi hesabı ve kullanım koşullarına bağlıdır.
