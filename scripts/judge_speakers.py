import os, sys, json, time
from google import genai
from google.genai import types
client = genai.Client(vertexai=True, project=os.environ["GOOGLE_CLOUD_PROJECT"], location="global")
clip, out, offset = sys.argv[1], sys.argv[2], float(sys.argv[3])
models = ["gemini-3.8-flash", "gemini-3.5-flash"]
prompt = """你是说话人分离(diarization)的人工裁判。请完整听这段中文访谈录音，回答:
1. 一共有几个不同的人在说话?分别是什么角色(如 主持人/嘉宾)?依据是什么(音色、内容)?
2. 按时间顺序列出每一个说话人轮次(turn):start/end 用 mm:ss 表示,speaker 用角色名,text 给出该轮次的原话(逐字,尽量完整)。
   注意:哪怕只有"对""嗯"这种一两个字的短应答,只要换了人,也要单独列为一个轮次。
只输出 JSON: {"num_speakers": int, "speakers": [{"role": str, "evidence": str}], "turns": [{"start": "mm:ss", "end": "mm:ss", "speaker": str, "text": str}]}"""
audio = types.Part.from_bytes(data=open(clip, "rb").read(), mime_type="audio/wav")
for m in models:
    t = time.time()
    try:
        r = client.models.generate_content(model=m, contents=[audio, prompt],
            config=types.GenerateContentConfig(audio_timestamp=True, response_mime_type="application/json", temperature=0.1))
        d = json.loads(r.text)
        d["_meta"] = {"model": m, "elapsed_s": round(time.time() - t, 1), "clip": clip, "offset_s": offset}
        json.dump(d, open(out, "w"), ensure_ascii=False, indent=1)
        print("OK", m, "speakers:", d["num_speakers"], "turns:", len(d["turns"]), round(time.time() - t, 1), "s")
        break
    except Exception as e:
        print("ERR", m, type(e).__name__, str(e)[:200])
