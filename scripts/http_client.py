import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

def request_prediction(url: str = "http://127.0.0.1:8080", x1: float = 1.0, x2: float = 1.0) -> dict:
    data = json.dumps({"x1": x1, "x2": x2}).encode("utf-8")
    req = urllib.request.Request(url + "/predict", data=data, headers={"Content-Type": "application/json"}, method="POST")
    
    start_time = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            
            # Проверяем наличие заголовка X-Request-ID (Nginx его добавляет)
            req_id = resp.headers.get("X-Request-ID")
            
            return {
                "ok": True, 
                "version": body.get("model_version"), 
                "status": resp.status,
                "ms": elapsed_ms,
                "request_id": req_id
            }
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return {"ok": False, "error": str(e), "status": getattr(e, 'code', 500), "ms": elapsed_ms}

def summary(results: list) -> dict:
    total = len(results)
    successes = [r for r in results if r["ok"]]
    errors = total - len(successes)
    
    by_version = {}
    by_version_metrics = {}
    unique_request_ids = set()
    model_version_missing = 0
    
    for r in successes:
        v = r.get("version")
        if not v:
            model_version_missing += 1
            v = "unknown"
            
        by_version[v] = by_version.get(v, 0) + 1
        
        if v not in by_version_metrics:
            by_version_metrics[v] = {"ms_list": [], "errors": 0}
        
        by_version_metrics[v]["ms_list"].append(r["ms"])
        if r.get("request_id"):
            unique_request_ids.add(r["request_id"])

    # Считаем процентили (упрощенно)
    def calc_percentile(data, p):
        if not data: return 0
        data = sorted(data)
        idx = int(len(data) * p / 100)
        return round(data[min(idx, len(data)-1)], 2)

    # Форматируем by_version_metrics
    final_metrics = {}
    for v, data in by_version_metrics.items():
        final_metrics[v] = {
            "requests": len(data["ms_list"]),
            "errors": data["errors"],
            "p95_ms": calc_percentile(data["ms_list"], 95),
            "error_rate": data["errors"] / len(data["ms_list"]) if data["ms_list"] else 0
        }

    all_ms = [r["ms"] for r in successes]
    
    return {
        "requests": total,
        "success": len(successes),
        "errors": errors,
        "by_version": by_version,
        "by_version_metrics": final_metrics,
        "p50_ms": calc_percentile(all_ms, 50),
        "p95_ms": calc_percentile(all_ms, 95),
        "p99_ms": calc_percentile(all_ms, 99),
        "unique_request_ids": len(unique_request_ids),
        "model_version_missing": model_version_missing,
        "error_rate": errors / total if total else 0
    }
