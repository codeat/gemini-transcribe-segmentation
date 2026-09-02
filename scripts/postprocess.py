#!/usr/bin/env python3
"""Gemini 3.5 Transcribe 结果后处理：把"按说话人轮次"的长段切成适合小屏显示的短句。
输入: generate_content 返回的 JSON (model_dump)，需开启 diarization=True, word_timestamp=True。
两级处理:
  Pass A (纯本地, 零成本): 利用词级时间戳，在 停顿>=GAP秒 / 句末标点 / 超过MAX字 处切分。
  Pass B (可选, --rerun): 对时长 > LONG 秒的轮次，截取该窗口音频单独再调一次模型做说话人分离，
          用重跑结果替换原长段（两人对话场景把局部标签映射回全局标签）。
用法: python3 postprocess.py result.json [--audio 原音频.wav --rerun] [--gap 0.6] [--max 60] [--long 90] [--out out.json]
"""
import json, re, argparse, subprocess, tempfile, os
END=set("。！？!?…"); CLAUSE=set("，,、；;：:")
def sec(s): return float(s.rstrip("s"))
def is_cjk(t): return re.search(r"[一-鿿]", t) is not None
def join(words): return ("" if is_cjk("".join(w["word"] for w in words)) else " ").join(w["word"] for w in words)
def load_turns(d):
    turns=[]
    for p in d["candidates"][0]["content"]["parts"]:
        at=p.get("audio_transcription",{}); w=at.get("words") or []
        if not w: continue
        turns.append(dict(spk=at.get("speaker_label"), words=w, start=sec(w[0]["start_offset"]), end=sec(w[-1]["end_offset"]), text=p["text"]))
    return turns
def split_turn(t, gap, maxc, minc=8):
    out=[]; cur=[]
    def flush():
        if cur: out.append(dict(spk=t["spk"], start=sec(cur[0]["start_offset"]), end=sec(cur[-1]["end_offset"]), text=join(cur)))
        cur.clear()
    for w in t["words"]:
        if cur:
            g=sec(w["start_offset"])-sec(cur[-1]["end_offset"]); n=len(join(cur)); last=cur[-1]["word"].rstrip()[-1:]
            if (g>=gap and n>=minc) or (last in END and n>=minc) or (last=="." and n>=minc and not is_cjk(last)): flush()
            elif n>=maxc:
                k=max((j for j,x in enumerate(cur) if x["word"].rstrip()[-1:] in CLAUSE|END), default=None)
                if k is not None and k>=2: tail=cur[k+1:]; del cur[k+1:]; flush(); cur.extend(tail)
                else: flush()
        cur.append(w)
    flush(); return out
def rerun_window(audio, start, end, pad=2.0):
    from google import genai; from google.genai import types
    client=genai.Client(vertexai=True, project=os.environ["GOOGLE_CLOUD_PROJECT"], location="global")
    s=max(0,start-pad); fd,tmp=tempfile.mkstemp(suffix=".wav"); os.close(fd)
    subprocess.run(["ffmpeg","-v","error","-y","-i",audio,"-ss",str(s),"-t",str(end-s+pad),tmp],check=True)
    r=client.models.generate_content(model="gemini-3.5-transcribe-preview",
        contents=[types.Part.from_bytes(data=open(tmp,"rb").read(),mime_type="audio/wav")],
        config=types.GenerateContentConfig(audio_transcription_config=types.AudioTranscriptionConfig(diarization=True,word_timestamp=True)))
    os.unlink(tmp); d=r.model_dump(exclude_none=True,mode="json"); sub=load_turns(d)
    for t in sub:  # 时间轴平移回全局
        for w in t["words"]: w["start_offset"]=f"{sec(w['start_offset'])+s:.3f}s"; w["end_offset"]=f"{sec(w['end_offset'])+s:.3f}s"
        t["start"]+=s; t["end"]+=s
    return sub
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("json"); ap.add_argument("--audio"); ap.add_argument("--rerun",action="store_true")
    ap.add_argument("--gap",type=float,default=0.6); ap.add_argument("--max",type=int,default=60); ap.add_argument("--long",type=float,default=90); ap.add_argument("--out")
    a=ap.parse_args(); d=json.load(open(a.json)); turns=load_turns(d); rerun_n=0
    if a.rerun and a.audio:
        fixed=[]
        for i,t in enumerate(turns):
            if t["end"]-t["start"]<=a.long: fixed.append(t); continue
            sub=rerun_window(a.audio,t["start"],t["end"]); rerun_n+=1
            sub=[x for x in sub if x["end"]>t["start"]+0.5 and x["start"]<t["end"]-0.5]   # 去掉 pad 带进来的边缘
            labels={x["spk"] for x in sub}
            if len(labels)==2 and i>0:   # 两人对话: 与前一段说话人不同者为"另一人"
                prev=turns[i-1]["spk"]; other=[s for s in {turns[j]["spk"] for j in range(len(turns))} if s!=prev]
                # 重跑首段紧接前一段, 通常是回应者(=原长段 spk); 用原长段标签给首段, 另一标签给另一人
                m={sub[0]["spk"]: t["spk"]}; o=[l for l in labels if l!=sub[0]["spk"]][0]; m[o]=(other[0] if other else o)
                for x in sub: x["spk"]=m[x["spk"]]
            fixed.extend(sub)
        turns=fixed
    segs=[]
    for t in turns: segs.extend(split_turn(t,a.gap,a.max))
    out={"segments":segs,"stats":{"turns_in":len(load_turns(d)),"segments_out":len(segs),"reran_long_turns":rerun_n,
         "max_chars":max(len(s["text"]) for s in segs),"max_dur_s":round(max(s["end"]-s["start"] for s in segs),1)}}
    if a.out: json.dump(out,open(a.out,"w"),ensure_ascii=False,indent=1)
    print(json.dumps(out["stats"],ensure_ascii=False))
    for s in segs: print(f"[{int(s['start']//60)}:{int(s['start']%60):02d}] {s['spk']} {s['text']}")
if __name__=="__main__": main()
