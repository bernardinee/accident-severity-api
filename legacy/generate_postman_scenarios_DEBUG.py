"""
Generate Postman scenarios - ADAPTIVE VERSION
Works with any response format
"""

import numpy as np
import json
from scipy.stats import norm

def generate_scenario(name, peak_g, duration_ms, axis='ax', crash_time_sec=2.5):
    """Generate realistic crash scenario"""
    ax = np.random.normal(0, 0.3, 500)
    ay = np.random.normal(0, 0.3, 500)
    az = np.random.normal(9.8, 0.2, 500)
    gx = np.random.normal(0, 2.0, 500)
    gy = np.random.normal(0, 2.0, 500)
    gz = np.random.normal(0, 2.0, 500)
    
    if peak_g > 2.0:
        duration_samples = int(duration_ms / 1000 * 100)
        sigma = duration_samples / 6
        
        t = np.arange(-3*sigma, 3*sigma)
        pulse = peak_g * norm.pdf(t, 0, sigma) / norm.pdf(0, 0, sigma)
        
        crash_center = int(crash_time_sec * 100)
        pulse_start = crash_center - len(pulse) // 2
        pulse_end = pulse_start + len(pulse)
        
        if pulse_start < 0:
            pulse = pulse[abs(pulse_start):]
            pulse_start = 0
        if pulse_end > 500:
            pulse = pulse[:500 - pulse_start]
            pulse_end = 500
        
        if axis == 'ax':
            ax[pulse_start:pulse_end] += pulse
            ay[pulse_start:pulse_end] += pulse * 0.2
            az[pulse_start:pulse_end] += pulse * 0.15
            gy[pulse_start:pulse_end] += pulse * 5.0
        elif axis == 'ay':
            ay[pulse_start:pulse_end] += pulse
            ax[pulse_start:pulse_end] += pulse * 0.25
            az[pulse_start:pulse_end] += pulse * 0.15
            gx[pulse_start:pulse_end] += pulse * 8.0
        elif axis == 'az':
            az[pulse_start:pulse_end] += pulse
            ax[pulse_start:pulse_end] += pulse * 0.3
            ay[pulse_start:pulse_end] += pulse * 0.3
            gx[pulse_start:pulse_end] += pulse * 15.0
            gy[pulse_start:pulse_end] += pulse * 10.0
    
    ax += np.random.normal(0, 0.05, 500)
    ay += np.random.normal(0, 0.05, 500)
    az += np.random.normal(0, 0.05, 500)
    gx += np.random.normal(0, 0.5, 500)
    gy += np.random.normal(0, 0.5, 500)
    gz += np.random.normal(0, 0.5, 500)
    
    return {
        "name": name,
        "ax": ax.round(6).tolist(),
        "ay": ay.round(6).tolist(),
        "az": az.round(6).tolist(),
        "gx": gx.round(6).tolist(),
        "gy": gy.round(6).tolist(),
        "gz": gz.round(6).tolist(),
        "expected_class": 0 if peak_g < 2 else (1 if peak_g < 7 else 2),
        "expected_severity": "Normal" if peak_g < 2 else ("Moderate" if peak_g < 7 else "Severe"),
        "peak_g": round(peak_g, 2)
    }

scenarios = [
    generate_scenario("Normal - Smooth Highway", peak_g=0.5, duration_ms=0),
    generate_scenario("Normal - City Traffic", peak_g=1.2, duration_ms=0),
    generate_scenario("Normal - Lane Change", peak_g=0.8, duration_ms=0, axis='ay'),
    generate_scenario("Normal - Speed Bump 10kmh", peak_g=1.5, duration_ms=200, axis='az'),
    generate_scenario("Moderate - Emergency Brake", peak_g=2.5, duration_ms=1500, axis='ax'),
    generate_scenario("Moderate - Sharp Turn", peak_g=3.2, duration_ms=800, axis='ay'),
    generate_scenario("Moderate - Pothole", peak_g=4.5, duration_ms=150, axis='az'),
    generate_scenario("Moderate - Lane Swerve", peak_g=3.8, duration_ms=600, axis='ay'),
    generate_scenario("Moderate - Speed Bump 40kmh", peak_g=5.2, duration_ms=250, axis='az'),
    generate_scenario("Severe - Frontal Low Speed", peak_g=12.0, duration_ms=120, axis='ax'),
    generate_scenario("Severe - Frontal High Speed", peak_g=22.0, duration_ms=100, axis='ax'),
    generate_scenario("Severe - Side T-Bone", peak_g=16.5, duration_ms=90, axis='ay'),
    generate_scenario("Severe - Pole Impact", peak_g=35.0, duration_ms=60, axis='ay'),
    generate_scenario("Severe - Rear End", peak_g=10.5, duration_ms=150, axis='ax'),
    generate_scenario("Severe - Rollover", peak_g=8.5, duration_ms=300, axis='az'),
    generate_scenario("Edge - Below Threshold", peak_g=1.9, duration_ms=500, axis='ax'),
    generate_scenario("Edge - Just Above Threshold", peak_g=2.1, duration_ms=500, axis='ax'),
    generate_scenario("Edge - Moderate-Severe Boundary", peak_g=6.9, duration_ms=200, axis='ax'),
    generate_scenario("Edge - Just Severe", peak_g=7.1, duration_ms=200, axis='ax'),
]

# ADAPTIVE TEST SCRIPT - Works with any response format
test_script = """
pm.test("Status code is 200", function () {
    pm.response.to.have.status(200);
});

pm.test("Response is valid JSON", function () {
    pm.response.to.be.json;
    var jsonData = pm.response.json();
    
    // Log entire response to see what we're getting
    console.log("Full API Response:", JSON.stringify(jsonData, null, 2));
});

pm.test("No API errors", function () {
    var jsonData = pm.response.json();
    if (jsonData.error) {
        console.log("API Error:", jsonData.error);
        pm.expect.fail("API returned error: " + jsonData.error);
    }
});

pm.test("Response has accident confirmation", function () {
    var jsonData = pm.response.json();
    pm.expect(jsonData).to.have.property('accident_confirmed');
    console.log("Accident Confirmed:", jsonData.accident_confirmed);
});

// Log summary
var jsonData = pm.response.json();
console.log("=".repeat(60));
console.log("Test:", pm.info.requestName);
console.log("All Fields:", Object.keys(jsonData).join(", "));
console.log("=".repeat(60));
"""

collection = {
    "info": {
        "name": "Accident Detection API - Debug Version",
        "description": "Adaptive tests to discover actual API response format",
        "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
    },
    "variable": [
        {
            "key": "base_url",
            "value": "https://accident-severity-api-production.up.railway.app",
            "type": "string"
        }
    ],
    "item": [
        {
            "name": "0. Health Check",
            "request": {
                "method": "GET",
                "header": [],
                "url": {
                    "raw": "{{base_url}}/health",
                    "host": ["{{base_url}}"],
                    "path": ["health"]
                }
            },
            "event": [{
                "listen": "test",
                "script": {
                    "exec": [
                        "pm.test(\"Status code is 200\", function () {",
                        "    pm.response.to.have.status(200);",
                        "});",
                        "var jsonData = pm.response.json();",
                        "console.log(\"Health Response:\", JSON.stringify(jsonData, null, 2));"
                    ],
                    "type": "text/javascript"
                }
            }]
        }
    ]
}

for i, scenario in enumerate(scenarios, 1):
    payload = {
        "ax": scenario['ax'],
        "ay": scenario['ay'],
        "az": scenario['az'],
        "gx": scenario['gx'],
        "gy": scenario['gy'],
        "gz": scenario['gz']
    }
    
    item = {
        "name": f"{i}. {scenario['name']} ({scenario['peak_g']}g)",
        "request": {
            "method": "POST",
            "header": [{"key": "Content-Type", "value": "application/json"}],
            "body": {
                "mode": "raw",
                "raw": json.dumps(payload)
            },
            "url": {
                "raw": "{{base_url}}/predict",
                "host": ["{{base_url}}"],
                "path": ["predict"]
            }
        },
        "event": [{
            "listen": "test",
            "script": {
                "exec": test_script.strip().split('\n'),
                "type": "text/javascript"
            }
        }]
    }
    collection["item"].append(item)

output_file = 'Postman_Tests_DEBUG.json'
with open(output_file, 'w') as f:
    json.dump(collection, f, indent=2)

print("=" * 70)
print("✓ DEBUG COLLECTION GENERATED")
print("=" * 70)
print(f"\nFile: {output_file}")
print("\nThis collection will:")
print("  1. Show you the EXACT response format from your API")
print("  2. Log all fields in the Console")
print("  3. Help us create correct test assertions")
print("\nTO USE:")
print("  1. Import Postman_Tests_DEBUG.json")
print("  2. Run collection")
print("  3. Open Console (View → Show Postman Console)")
print("  4. Copy the 'Full API Response' from Console")
print("  5. Share it with me")
print("=" * 70)