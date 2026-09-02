import os, sys, json, time, argparse
from google import genai
from google.genai import types
ap = argparse.ArgumentParser()
ap.add_argument("audio"); ap.add_argument("out")
ap.add_argument("--loc", default="global")
ap.add_argument("--model", default="gemini-3.5-transcribe-preview")
ap.add_argument("--lang", nargs="*", default=None)
ap.add_argument("--no-wordts", action="store_true")
ap.add_argument("--no-diar", action="store_true")
ap.add_argument("--prompt", default=None)
ap.add_argument("--mode", default="VERBATIM"); ap.add_argument("--maxtok", type=int, default=None)
a = ap.parse_args()
client = genai.Client(vertexai=True, project=os.environ["GOOGLE_CLOUD_PROJECT"], location=a.loc)
parts = [types.Part.from_bytes(data=open(a.audio,"rb").read(), mime_type="audio/wav")]
if a.prompt: parts.append(types.Part.from_text(text=a.prompt))
cfg = types.AudioTranscriptionConfig(mode=a.mode, diarization=not a.no_diar, word_timestamp=not a.no_wordts, language_codes=a.lang)
t=time.time()
try:
    r = client.models.generate_content(model=a.model, contents=parts,
        config=types.GenerateContentConfig(audio_transcription_config=cfg, max_output_tokens=a.maxtok))
    d = r.model_dump(exclude_none=True, mode="json"); d.pop("sdk_http_response", None)
    d["_meta"] = {"args": vars(a), "elapsed_s": round(time.time()-t,1)}
    json.dump(d, open(a.out,"w"), ensure_ascii=False, indent=1)
    n = len(d["candidates"][0]["content"]["parts"])
    print("OK", a.out, "segments:", n, "elapsed:", d["_meta"]["elapsed_s"], "s", "usage:", d.get("usage_metadata",{}).get("total_token_count"))
except Exception as e:
    print("ERR", a.out, type(e).__name__, str(e)[:800])
