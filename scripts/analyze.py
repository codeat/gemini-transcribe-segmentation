import json, sys, re
def sec(s): return float(s.rstrip("s"))
def load(p):
    d=json.load(open(p)); segs=[]
    for part in d["candidates"][0]["content"]["parts"]:
        at=part.get("audio_transcription",{}); w=at.get("words",[])
        st=sec(w[0]["start_offset"]) if w else None; en=sec(w[-1]["end_offset"]) if w else None
        segs.append(dict(spk=at.get("speaker_label"), text=part["text"], start=st, end=en, words=w))
    return d, segs
def fmt(t): return "%d:%02d"%(t//60,t%60) if t is not None else "?"
if __name__=="__main__":
    p=sys.argv[1]; d,segs=load(p); zh=re.search(r"[一-鿿]", segs[0]["text"]) is not None
    lens=[len(s["text"]) for s in segs]; durs=[(s["end"]-s["start"]) for s in segs if s["start"] is not None]
    print(f"{p}: segments={len(segs)} elapsed={d['_meta']['elapsed_s']}s")
    print(f"  chars/seg: mean={sum(lens)/len(lens):.0f} max={max(lens)} >200chars:{sum(l>200 for l in lens)} >500chars:{sum(l>500 for l in lens)}")
    if durs: print(f"  dur/seg: mean={sum(durs)/len(durs):.1f}s max={max(durs):.1f}s >30s:{sum(x>30 for x in durs)} >60s:{sum(x>60 for x in durs)}")
    from collections import Counter; print("  speakers:", Counter(s["spk"] for s in segs))
    # top-5 longest with speech rate
    top=sorted([s for s in segs if s["start"] is not None], key=lambda s:-(s["end"]-s["start"]))[:6]
    for s in top:
        dur=s["end"]-s["start"]; rate=len(s["text"])/dur if dur else 0
        print(f"  [{fmt(s['start'])}-{fmt(s['end'])}] {s['spk']} dur={dur:.0f}s chars={len(s['text'])} rate={rate:.1f} chars/s  {s['text'][:60]}...")
