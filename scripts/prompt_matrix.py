import os, sys, json, time
from google import genai
from google.genai import types
client = genai.Client(vertexai=True, project=os.environ["GOOGLE_CLOUD_PROJECT"], location="global")
clip="clips/zh_13m_3min.wav"
audio = types.Part.from_bytes(data=open(clip,"rb").read(), mime_type="audio/wav")
cfg = types.AudioTranscriptionConfig(mode="VERBATIM", diarization=True, word_timestamp=True)
ZH="请按句子分段输出，每段不超过40个字，每句单独成段。"
EN="Split the transcript into short segments of at most 40 characters, one sentence per segment."
variants = {
 "baseline": dict(contents=[audio], config=types.GenerateContentConfig(audio_transcription_config=cfg)),
 "text_after": dict(contents=[audio, types.Part.from_text(text=ZH)], config=types.GenerateContentConfig(audio_transcription_config=cfg)),
 "text_before": dict(contents=[types.Part.from_text(text=ZH), audio], config=types.GenerateContentConfig(audio_transcription_config=cfg)),
 "system_instr": dict(contents=[audio], config=types.GenerateContentConfig(audio_transcription_config=cfg, system_instruction=ZH)),
 "english_after": dict(contents=[audio, types.Part.from_text(text=EN)], config=types.GenerateContentConfig(audio_transcription_config=cfg)),
 "two_turns": dict(contents=[types.Content(role="user", parts=[types.Part.from_text(text=ZH)]), types.Content(role="user", parts=[audio])], config=types.GenerateContentConfig(audio_transcription_config=cfg)),
 "no_diar_text_after": dict(contents=[audio, types.Part.from_text(text=ZH)], config=types.GenerateContentConfig(audio_transcription_config=types.AudioTranscriptionConfig(mode="VERBATIM"))),
 "no_cfg_text_after": dict(contents=[audio, types.Part.from_text(text=ZH)], config=None),
}
name=sys.argv[1]; v=variants[name]; t=time.time()
try:
    r = client.models.generate_content(model="gemini-3.5-transcribe-preview", **v)
    d = r.model_dump(exclude_none=True, mode="json"); d.pop("sdk_http_response", None)
    json.dump(d, open(f"results/pm_{name}.json","w"), ensure_ascii=False, indent=1)
    parts=d["candidates"][0]["content"]["parts"]; u=d.get("usage_metadata",{})
    print("OK", name, "parts", len(parts), "max_chars", max(len(p.get("text","")) for p in parts), "prompt_tok", u.get("prompt_token_count"), "out_tok", u.get("candidates_token_count"), round(time.time()-t,1),"s")
except Exception as e:
    print("ERR", name, type(e).__name__, str(e)[:300])
