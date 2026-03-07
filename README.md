# Weighbridge Monitoring System - API Documentation

## نظرة عامة (Overview)

سیستم مانیتورینگ باسکول با قابلیت تشخیص خودکار وضعیت خودروها و اعتبارسنجی تصاویر

## ساختار پروژه (Project Structure)

```
├── src/
│   ├── api.py                      # REST API با پشتیبانی base64
│   ├── simple_validator.py         # Validation logic اصلی
│   ├── object_detector.py          # YOLO object detection
│   └── image_utils.py              # Image processing utilities
├── configs/
│   ├── wb1_single_cam_roi.json    # تنظیمات باسکول 1
│   └── wb3_single_cam_roi.json    # تنظیمات باسکول 3 (برای نرم‌افزار ID=2)
├── weights/
│   └── best.pt                     # YOLOv11 model weights
├── tools/
│   ├── weighbridge_config_simple.html    # ابزار تنظیم polygon‌ها
│   ├── weighbridge_config_master.html
│   └── weighbridge_config_master1.html
├── batch_simple.py                 # اسکریپت پردازش دسته‌ای
├── test_api.py                     # اسکریپت تست API
├── docker-compose.yml              # Docker orchestration
└── Dockerfile                      # Container configuration

```

## نصب و راه‌اندازی (Installation)

### روش 1: استفاده از Docker (توصیه می‌شود)

```bash
# ساخت و اجرای container
docker-compose up -d --build

# مشاهده لاگ‌ها
docker-compose logs -f

# توقف container
docker-compose down
```

API روی پورت 4001 در دسترس خواهد بود: `http://localhost:4001`

### روش 2: اجرای مستقیم

```bash
# نصب dependencies
pip install -r requirements.txt

# اجرای API
python3 src/api.py
```

## API Endpoints

### 1. Health Check

**GET** `/health`

```bash
curl http://localhost:4001/health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "weighbridge-validation-api",
  "version": "2.0.0"
}
```

### 2. Image Validation

**POST** `/validate`

**Request:**
```json
{
  "weighbridge_id": 1,
  "image_base64": "base64_encoded_image_string"
}
```

**Response:**
```json
{
  "valid": true,
  "description": "VALID - Vehicle properly positioned in yellow area",
  "processed_image_base64": "base64_encoded_annotated_image_string"
}
```

**نکات مهم:**
- `weighbridge_id`: فقط 1 یا 3 قابل قبول است
  - **1** → استفاده از کانفیگ `wb1_single_cam_roi.json`
  - **2** → استفاده از کانفیگ `wb3_single_cam_roi.json` (wb3)
- `image_base64`: تصویر به فرمت base64 (با یا بدون header)
- `processed_image_base64`: تصویر پردازش شده با annotations

## تست با Python Script

```bash
# تست با تصویر نمونه
python test_api.py baskol-test/bask2_up_20251202_110500.jpg 2

# استفاده کامل
python test_api.py <image_path> [weighbridge_id] [api_url]
```

## تست با cURL

```bash
# Encode image to base64
IMAGE_BASE64=$(base64 -w 0 baskol-test/bask2_up_20251202_110500.jpg)

# Send request
curl -X POST http://localhost:4001/validate \
  -H "Content-Type: application/json" \
  -d "{
    \"weighbridge_id\": 2,
    \"image_base64\": \"$IMAGE_BASE64\"
  }"
```

## تست با Postman

1. ایجاد یک **POST** request به `http://localhost:4001/validate`
2. در تب **Body**، گزینه **raw** و **JSON** را انتخاب کنید
3. محتوای زیر را وارد کنید:

```json
{
  "weighbridge_id": 2,
  "image_base64": "<YOUR_BASE64_IMAGE_STRING>"
}
```

4. دکمه **Send** را بزنید

## قوانین اعتبارسنجی (Validation Rules)

API بر اساس قوانین زیر تصاویر را اعتبارسنجی می‌کند:

### 1. تشخیص خطوط زرد
- Threshold: 1% از سطح polygon
- اگر خطوط زرد قابل رؤیت باشد → استفاده از **scale area (yellow)**
- اگر خطوط زرد نامرئی باشد → استفاده از **prohibited zone (red)**

### 2. تعداد خودرو
- باید دقیقاً **1 خودرو** در تصویر باشد
- خودروهای مجاز: car, truck, bus

### 3. اشیاء ممنوعه
- موتورسیکلت، دوچرخه → **INVALID**
- شخص (person) → **INVALID**

### 4. موقعیت خودرو

#### برای Yellow Scale Area:
- هیچ کدام از گوشه‌های bounding box نباید **بیش از 60 پیکسل** خارج از polygon باشد
- اگر 2 یا بیشتر گوشه بیش از 60px خارج باشد → **INVALID**

#### برای Red Prohibited Zone - خودروهای کوچک (<92%):
- هیچ کدام از گوشه‌های bounding box نباید **بیش از 40 پیکسل** خارج از polygon باشد
- اگر 2 یا بیشتر گوشه بیش از 40px خارج باشد → **INVALID**

#### برای Red Prohibited Zone - خودروهای بزرگ (≥92%):
- مرکز خودرو باید در **margin عمودی 200 پیکسلی** قرار گیرد
- margin در مرکز prohibited zone قرار دارد
- اگر مرکز خودرو خارج از margin باشد → **INVALID**

## ابزار تنظیم Polygon‌ها

برای تنظیم مناطق (scale area و prohibited zone):

```bash
# باز کردن ابزار HTML در مرورگر
firefox weighbridge_config_simple.html
```

این ابزار امکان می‌دهد:
- رسم polygon‌های زرد و قرمز روی تصویر 1920x1080
- ذخیره مختصات در فایل JSON
- تست مستقیم validation

## فایل‌های کانفیگ

### نمونه `wb1_single_cam_roi.json`:

```json
{
  "weighbridge_name": "Baskol 1",
  "camera": "UP",
  "areas": {
    "scale_area": {
      "points": [[x1, y1], [x2, y2], ...]
    },
    "prohibited_zone": {
      "points": [[x1, y1], [x2, y2], ...]
    }
  }
}
```

## خطاها و عیب‌یابی

### خطای "Config file not found"
- مطمئن شوید فایل‌های `wb1_single_cam_roi.json` و `wb3_single_cam_roi.json` در پوشه `configs/` موجود هستند

### خطای "Invalid base64 image"
- مطمئن شوید تصویر به درستی به base64 تبدیل شده است
- API از header دار و بدون header پشتیبانی می‌کند

### خطای "Model not found"
- مطمئن شوید فایل `weights/best.pt` موجود است
- اگر ندارید، مدل YOLO را دانلود کنید

## Performance Notes

- تصاویر 1920x1080 پردازش می‌شوند (بدون cropping)
- زمان پردازش: ~1-3 ثانیه بر روی CPU
- حداکثر حجم تصویر: 32MB

## نمونه کد برای نرم‌افزار

### C# Example

```csharp
using System.Net.Http;
using System.Text;

public async Task<ValidationResult> ValidateImage(int weighbridgeId, byte[] imageBytes)
{
    var base64Image = Convert.ToBase64String(imageBytes);
    
    var payload = new {
        weighbridge_id = weighbridgeId,
        image_base64 = base64Image
    };
    
    var json = JsonConvert.SerializeObject(payload);
    var content = new StringContent(json, Encoding.UTF8, "application/json");
    
    var response = await httpClient.PostAsync("http://localhost:4001/validate", content);
    var result = await response.Content.ReadAsStringAsync();
    
    return JsonConvert.DeserializeObject<ValidationResult>(result);
}
```

### Python Example

```python
import requests
import base64

def validate_image(weighbridge_id, image_path):
    # Encode image
    with open(image_path, 'rb') as f:
        image_base64 = base64.b64encode(f.read()).decode('utf-8')
    
    # Prepare request
    payload = {
        "weighbridge_id": weighbridge_id,
        "image_base64": image_base64
    }
    
    # Send request
    response = requests.post(
        'http://localhost:4001/validate',
        json=payload
    )
    
    return response.json()
```

## License

Internal use only - Weighbridge Monitoring System
