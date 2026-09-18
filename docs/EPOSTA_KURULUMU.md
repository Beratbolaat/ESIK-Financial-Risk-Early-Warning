# Şirket raporunu e-postayla gönderme

Şirket Detayı → Analiz raporunu e-postayla gönder → raporu ve alıcıyı kontrol et → E-postayla gönder.

Sohbet soruları veya sayfa yenileme e-posta göndermez. Gönderilen metin model çıktıları ve SHAP tablosundan hazırlanır; OpenAI'ye yeni sayı ürettirilmez. Ham ve kalibre olasılık ayrıdır. Rapor gerçekleşmiş bireysel iflas etiketini içermez. Güncel bir gerçek şirkete ait rapor değildir.

## Başka bilgisayarda kurulum

1. n8n'e `n8n/esik-email-workflow.json` dosyasını yeni iş akışı olarak içe aktar.
2. **Rapor Webhook** düğümünde mevcut EŞİK Header Auth bağlantını seç. Header adı `X-ESIK-WEBHOOK-TOKEN` olmalıdır.
3. **Rapor ve Alıcı Kontrolü** içindeki `sender@example.com` yer tutucusunu kendi gönderen/alıcı adresinle değiştir. Bu sürüm raporu yalnız o adrese gönderir.
4. **Gmail ile Rapor Gönder** düğümünde SMTP bağlantını seç. Gmail: `smtp.gmail.com`, port `465`, SSL/TLS açık. Parola Python'a veya workflow dosyasına yazılmaz; n8n bağlantısında saklanır.
5. Akışı Publish ile etkinleştir. `.streamlit/secrets.toml` dosyasına aşağıdaki alanları ekle. Mevcut sohbet ve ses alanlarını koru.

```toml
N8N_EMAIL_WEBHOOK_URL = "http://127.0.0.1:5678/webhook/esik-report-email"
ESIK_REPORT_EMAIL = "sender@example.com"
# N8N_WEBHOOK_TOKEN, n8n Header Auth ile aynı mevcut anahtar olmalı.
```

`secrets.toml` ve SMTP parolası paylaşım ZIP'ine dahil edilmez.

## Akış ve sonucu yorumlama

Streamlit raporu hazırlar → kimlik doğrulamalı webhook → alıcı ve rapor kontrolü → SMTP gönderimi → HTTP yanıtı.

Başarı yalnız SMTP sunucusunun alıcıyı ve iletiyi kabul ettiğini gösterir; gelen kutusuna teslim garantisi değildir. Spam klasörüne düşebilir. Hata/timeout sonucu belirsizse otomatik tekrar gönderilmez; önce Gmail Gönderilmiş klasörü ve n8n çalışması kontrol edilir. Aynı rapor başarılı gönderimden sonra aynı Streamlit oturumunda tekrar gönderilemez. Yeni oturumlar arasında kalıcı tekrar önleme bu demo kapsamına dahil değildir.

Şirket raporu yalın metin olarak e-postanın gövdesine eklenir; ek dosya gerekmez.

Resmî SMTP düğümü: [n8n Send Email](https://github.com/n8n-io/n8n-docs/blob/main/docs/integrations/builtin/core-nodes/n8n-nodes-base.sendemail.md).
