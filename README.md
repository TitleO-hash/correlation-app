# Correlation & Buffer Calculator

Web app สำหรับวิเคราะห์ Correlation Matrix และคำนวณ Portfolio Buffer / Leverage Ratio

## Features
- ดึงข้อมูลราคาจาก Yahoo Finance อัตโนมัติ
- รองรับ 4 กลุ่มสินทรัพย์: FX, Stock, Commodity, Crypto
- คำนวณ Correlation Matrix (Daily Returns-based)
- คำนวณ Portfolio VaR (99%) และ Buffer ที่ต้องเผื่อ
- เปรียบเทียบ Leverage Ratio 3 scenario
- Risk Contribution breakdown รายกลุ่ม

## วิธีใช้งาน
1. ใส่ symbol ในแต่ละกลุ่ม (ซ้ายมือ)
2. เลือกช่วงเวลา (1Y / 3Y / 5Y)
3. ใส่ทุน ($)
4. กด **Calculate**

## Symbols ที่รองรับ
| กลุ่ม | ตัวอย่าง |
|---|---|
| FX | EURUSD, GBPUSD, USDJPY, CHFJPY, EURGBP, USDCAD |
| Stock | NVDA, GOOG, TSLA, AAPL, MSFT, SPY, QQQ |
| Commodity | XAUUSD, XAGUSD, USOIL, NGAS |
| Crypto | BTCUSD, ETHUSD, XRPUSD, SOLUSD, BNBUSD |

## Deploy บน Streamlit Cloud

1. Fork/push repo นี้ขึ้น GitHub
2. ไปที่ [share.streamlit.io](https://share.streamlit.io)
3. กด **New app** → เลือก repo → `app.py` → Deploy

## โครงสร้างไฟล์
```
correlation_app/
├── app.py
├── requirements.txt
└── .streamlit/
    └── config.toml
```

## คณิตศาสตร์ที่ใช้
- **Correlation**: Pearson r บน Daily Returns
- **Portfolio σ**: √(wᵀ Σ w) โดย Σ = correlation × outer(σ, σ)
- **VaR 99%**: z × σ_p × Capital  (z = 2.326)
- **Max Leverage**: Capital / VaR

---
*ใช้สำหรับประกอบการตัดสินใจเท่านั้น ไม่ใช่คำแนะนำการลงทุน*
