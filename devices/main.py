import network
import urequests
from machine import Pin, ADC
from time import sleep, time
import dht
import gc
import json 

# ==========================================
# ⚙️ CLOUD CONFIGURATION
# ==========================================
WIFI_SSID
WIFI_PASSWORD

# 🔴 CONFIG: SUPABASE URLS
REALTIME_URL
HISTORY_URL

# 🔴 KEY
SUPABASE_KEY

# ==========================================
# 📡 WIFI CONNECTION
# ==========================================
def connect_wifi():
    wlan = network.WLAN(network.STA_IF)
    wlan.active(False)
    sleep(1)
    wlan.active(True)
    
    if not wlan.isconnected():
        print(f"📡 Connecting to {WIFI_SSID}...")
        wlan.connect(WIFI_SSID, WIFI_PASSWORD)
        timeout = 20
        while not wlan.isconnected() and timeout > 0:
            sleep(1)
            timeout -= 1
            
    if wlan.isconnected():
        print(f"✅ Connected! IP: {wlan.ifconfig()[0]}")
        return True
    else:
        print("❌ WiFi Failed")
        return False

# Connect First
while not connect_wifi():
    print("Retrying WiFi...")
    sleep(5)

# ==========================================
# 📟 SENSORS
# ==========================================
dht_sensor = dht.DHT22(Pin(4))
mq135 = ADC(Pin(34))
mq135.atten(ADC.ATTN_11DB)

# Helper Functions
def mq135_to_pm25(raw):
    return ((raw - 0) / (4095 - 0)) * 200

def calculate_aqi_pm25(pm25):
    if pm25 <= 12: return round((50/12)*pm25)
    if pm25 <= 35.4: return round(((49/23.3)*(pm25-12.1)) + 51)
    if pm25 <= 55.4: return round(((49/19.9)*(pm25-35.5)) + 101)
    return 150

# Headers
HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": "Bearer " + SUPABASE_KEY,
    "Content-Type": "application/json",
    "Prefer": "return=minimal" 
}

# Variables
sum_mq135 = 0
sum_temp = 0
sum_hum = 0
sum_aqi = 0
sample_count = 0
last_hourly_upload = time()
HOUR_INTERVAL = 3600

# ==========================================
# 🚀 MAIN LOOP
# ==========================================
print("🚀 Sensor Loop Started...")

while True:
    try:
        if not network.WLAN(network.STA_IF).isconnected():
            print("⚠️ WiFi dropped! Reconnecting...")
            connect_wifi()

        # 1. READ
        try:
            dht_sensor.measure()
            temp = dht_sensor.temperature()
            hum = dht_sensor.humidity()
        except OSError:
            temp, hum = 0, 0

        raw_air = mq135.read()
        pm25 = mq135_to_pm25(raw_air)
        aqi = calculate_aqi_pm25(pm25)

        # 2. ACCUMULATE
        sum_mq135 += raw_air
        sum_temp += temp
        sum_hum += hum
        sum_aqi += aqi
        sample_count += 1

        # 3. PUSH LIVE
        # Using simple types (int/float) to avoid database rejection
        payload_realtime = {
            "mq135": int(raw_air),
            "temperature": float(round(temp, 1)),
            "humidity": float(round(hum, 1)),
            "aqi": int(aqi)
        }
        
        print(f"☁️ Sending: {json.dumps(payload_realtime)}")

        try:
            resp = urequests.post(REALTIME_URL, json=payload_realtime, headers=HEADERS)
            
            # CHECK STATUS CODE
            if resp.status_code == 201:
                print("   ✅ Success: Uploaded to Supabase")
            else:
                # Print the actual error message from Supabase
                print(f"   ❌ FAILED (Status {resp.status_code}): {resp.text}")
                
            resp.close()
        except Exception as net_err:
            print(f"   ⚠ Network Error: {net_err}")

        # 4. HOURLY CHECK
        current_time = time()
        if (current_time - last_hourly_upload) >= HOUR_INTERVAL:
            print("\n⏳ Saving Hourly History...")
            if sample_count > 0:
                payload_history = {
                    "mq135": int(sum_mq135 / sample_count),
                    "temperature": float(round(sum_temp / sample_count, 2)),
                    "humidity": float(round(sum_hum / sample_count, 2)),
                    "aqi": int(sum_aqi / sample_count)
                }
                
                try:
                    resp_hist = urequests.post(HISTORY_URL, json=payload_history, headers=HEADERS)
                    if resp_hist.status_code == 201:
                        print("   ✅ History Saved.")
                    else:
                        print(f"   ❌ History Failed: {resp_hist.text}")
                    resp_hist.close()
                except Exception as e:
                    print(f"   ⚠ History Error: {e}")

            # Reset
            sum_mq135 = 0
            sum_temp = 0
            sum_hum = 0
            sum_aqi = 0
            sample_count = 0
            last_hourly_upload = time()

        gc.collect()
        sleep(2) 

    except Exception as e:
        print(f"⚠ Critical Error: {e}")
        sleep(5)
