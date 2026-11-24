"""Local source-indexed H3 graybox restyling using native ComfyUI nodes only."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shutil
import urllib.error
import urllib.request
from fractions import Fraction
from pathlib import Path

import av
import aiohttp
import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
COMFY = ROOT.parent / "ComfyUI"
SERVER = "http://127.0.0.1:8188"
SOURCE = ROOT / "outputs/stage_a/sample_66ad/original_video.mp4"
RUNS = ROOT / "outputs/h3_controlled_graybox"
PATCH = "minimax_h3_fun_controlnet_union_2.0_pruned_int8_convrot.safetensors"
PROMPT = (
    "A monochrome architectural graybox blockout animation of a third-person outdoor game scene. "
    "Everything is built from large simple untextured matte light-gray solid primitives, "
    "like an unfinished game level made of cubes, rectangular beams, cylinders and coarse rounded masses. "
    "A single human-shaped gray mannequin is in the foreground, with a simple solid head, "
    "a blocky torso and articulated solid limbs, holding a small axe made of a box head and straight handle. "
    "The mannequin slowly turns and moves along the path, following the exact movement and position "
    "in the control sequence. The camera follows behind at the same angle as the control sequence. "
    "The path is a smooth gray surface, rocks are large gray solid masses, "
    "fences are simple rectangular posts and horizontal beams, "
    "trees are cylinder trunks with large solid coarse canopy masses. "
    "Keep the same outdoor scene layout, subject scale, camera framing and foreground occlusions. "
    "All objects have the same pale gray clay material. Soft broad studio-like daylight, "
    "soft contact shadows and ambient occlusion make the volumes readable. "
    "Strongly simplify small details into a few large volumes. "
    "No realistic skin, no hair strands, no cloth folds, no surface textures, no leaves, no grass blades, "
    "no colors, no wireframe, no triangle mosaic, no voxel grid, no added furniture or indoor room. "
    "The graybox primitive appearance remains consistent throughout every frame."
)


def request(method, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(SERVER + path, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(exc.read().decode(errors="replace")) from exc


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Writer:
    def __init__(self, path, width, height, fps, crf=18):
        self.container = av.open(str(path), "w")
        self.stream = self.container.add_stream("libx264", rate=fps)
        self.stream.width, self.stream.height = width, height
        self.stream.pix_fmt = "yuv420p"
        self.stream.options = {"crf": str(crf), "preset": "fast"}
        self.fps, self.count = fps, 0

    def write(self, rgb):
        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb), format="rgb24")
        frame.pts, frame.time_base = self.count, Fraction(1, self.fps)
        for packet in self.stream.encode(frame):
            self.container.mux(packet)
        self.count += 1

    def close(self):
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()


def inspect(path):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        pts = [str(frame.pts * frame.time_base) for frame in container.decode(stream)]
        return {"path": str(path), "frames": len(pts), "fps": str(stream.average_rate),
                "width": stream.width, "height": stream.height, "pts_seconds": pts,
                "duration_seconds": str(stream.duration * stream.time_base) if stream.duration is not None else None}


def exact_timeline(probe, count, fps=60):
    return (probe["frames"] == count and Fraction(probe["fps"]) == fps
            and Fraction(probe["duration_seconds"]) == Fraction(count,fps)
            and [Fraction(p) for p in probe["pts_seconds"]] == [Fraction(i,fps) for i in range(count)])


def prepare(args):
    run = RUNS / args.name
    run.mkdir(parents=True, exist_ok=False)
    if (args.length - 5) % 17:
        raise ValueError("H3 length must be 17*n+5")
    control_name = f"h3_graybox_{args.name}_control.mp4"
    control = COMFY / "input" / control_name
    if control.exists():
        raise FileExistsError(control)
    writer = Writer(control, args.width, args.height, 24, crf=0)
    mapping, last, edge_density = [], None, []
    with av.open(str(SOURCE)) as container:
        stream = container.streams.video[0]
        if stream.average_rate != 60:
            raise ValueError("Expected the verified 60 FPS source")
        for index, frame in enumerate(container.decode(stream)):
            if index < args.start:
                continue
            if index >= args.start + args.length:
                break
            rgb = frame.to_ndarray(format="rgb24")
            resized = cv2.resize(rgb, (args.width, args.height), interpolation=cv2.INTER_AREA)
            if args.mode == "gray":
                gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
                gray = cv2.GaussianBlur(gray, (5, 5), 1.0)
                last = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            else:
                smooth = cv2.GaussianBlur(resized, (15, 15), 3.0)
                edges = cv2.Canny(smooth, 12, 30)
                edge_density.append(float(np.count_nonzero(edges)/edges.size))
                last = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            writer.write(last)
            mapping.append({"generated_index": len(mapping), "source_index": index,
                            "source_pts_seconds": str(frame.pts * frame.time_base), "padding": False})
    if last is None:
        raise ValueError("No source frames in selected range")
    real_count = len(mapping)
    while len(mapping) < args.length:
        writer.write(last)
        mapping.append({**mapping[-1], "generated_index": len(mapping), "padding": True})
    writer.close()
    if edge_density and np.mean(edge_density)<0.003:
        raise ValueError("Canny control is nearly empty; do not queue an unconditioned-looking clip")
    save_json(run / "run.json", {
        "schema": "h3.controlled-graybox/1", "source": str(SOURCE), "source_sha256": sha256(SOURCE),
        "start": args.start, "real_count": real_count, "length": args.length,
        "width": args.width, "height": args.height, "mode": args.mode,
        "control": control_name, "control_sha256": sha256(control), "mapping": mapping,
        "native_fps": 24, "delivery_fps": 60, "steps": args.steps,
        "turbo": not args.no_turbo, "strength": args.strength, "seed": args.seed,
        "control_preprocessing": {"gray_blur_sigma":1.0} if args.mode=="gray" else
            {"blur_sigma":3.0,"canny_low":12,"canny_high":30,"mean_edge_density":float(np.mean(edge_density))},
        "prompt": PROMPT, "status": "prepared",
        "limitations": ["Source index/PTS assignment is exact; generated geometry is soft-guided, not exact.",
                         "Native H3 sees contiguous source frames at 24 FPS; delivery uses the same indices at 60 FPS."]})
    print(f"Prepared {run}: {real_count} source frames + {args.length-real_count} padding", flush=True)


def graph(config, name):
    model = ["1", 0]
    result = {"1": {"class_type": "UNETLoader", "inputs": {
        "unet_name": "minimax_h3_fl2va_pruned_w6a8.safetensors", "weight_dtype": "default"}}}
    if config["turbo"]:
        result["2"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": model,
            "lora_name": "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors", "strength_model": 1.0}}
        model = ["2", 0]
    result.update({
        "3": {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": model, "shift_video": 12.0, "shift_audio": 3.0}},
        "4": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "type": "minimax", "device": "default"}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_video_vae_int8_convrot.safetensors"}},
        "6": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {"clip": ["4", 0], "vae": ["5", 0], "prompt": config["prompt"], "width": config["width"], "height": config["height"], "length": config["length"]}},
        "7": {"class_type": "LoadVideo", "inputs": {"file": config["control"]}},
        "8": {"class_type": "GetVideoComponents", "inputs": {"video": ["7", 0]}},
        "9": {"class_type": "ModelPatchLoader", "inputs": {"name": PATCH}},
        "10": {"class_type": "MiniMaxH3FunControlNetApply", "inputs": {"model": ["3", 0], "model_patch": ["9", 0], "vae": ["5", 0], "strength": config["strength"], "start_percent": 0.0, "end_percent": 1.0, "control_video": ["8", 0]}},
        "11": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "12": {"class_type": "BasicScheduler", "inputs": {"model": ["10", 0], "scheduler": "simple", "steps": config["steps"], "denoise": 1.0}},
        "13": {"class_type": "RandomNoise", "inputs": {"noise_seed": config["seed"]}},
        "14": {"class_type": "BasicGuider", "inputs": {"model": ["10", 0], "conditioning": ["6", 0]}},
        "15": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["13", 0], "guider": ["14", 0], "sampler": ["11", 0], "sigmas": ["12", 0], "latent_image": ["6", 1]}},
        "16": {"class_type": "VAEDecode", "inputs": {"samples": ["15", 0], "vae": ["5", 0]}},
        "17": {"class_type": "CreateVideo", "inputs": {"images": ["16", 0], "fps": 60.0, "codec": "auto", "bit_depth": 8, "color_space": "sRGB", "chroma_subsampling": "420"}},
        "18": {"class_type": "SaveVideo", "inputs": {"video": ["17", 0], "filename_prefix": f"h3_controlled_graybox/{name}", "format": "mp4"}},
    })
    if config.get("guide"):
        result["19"] = {"class_type": "LoadVideo", "inputs": {"file": config["guide"]}}
        result["20"] = {"class_type": "GetVideoComponents", "inputs": {"video": ["19", 0]}}
        result["21"] = {"class_type": "MiniMaxH3AddGuide", "inputs": {
            "positive": ["6", 0], "latent": ["6", 1], "vae": ["5", 0], "image": ["20", 0], "frame_idx": 0}}
        pad_w, pad_h = (-config["width"]) % 32, (-config["height"]) % 32
        if pad_w or pad_h:
            # Native H3 pads the target latent grid, but not keyframe latents.
            # Pad only guide pixels to that same grid; keep target size/indices.
            result["22"] = {"class_type": "EmptyMiniMaxH3LatentAV", "inputs": {
                "width":config["width"]+pad_w,"height":config["height"]+pad_h,"length":config["length"]}}
            result["23"] = {"class_type": "ImagePadForOutpaint", "inputs": {
                "image":["20",0],"left":0,"top":0,"right":pad_w,"bottom":pad_h,"feathering":0}}
            result["21"]["inputs"].update(latent=["22",0],image=["23",0])
        result["14"]["inputs"]["conditioning"] = ["21", 0]
    return result


def workflow(graph_data, definitions):
    nodes, links, by_id = [], [], {}
    positions = {1:(0,0),2:(0,180),3:(0,350),4:(0,550),5:(0,720),6:(700,550),
                 7:(350,0),8:(350,180),9:(350,360),10:(700,0),11:(1100,0),12:(1100,160),
                 13:(1100,340),14:(1100,520),15:(1450,300),16:(1800,300),17:(2150,300),18:(2500,300),
                 19:(0,1050),20:(350,1050),21:(1100,950),22:(700,1250),23:(700,950)}
    for order, (identity, item) in enumerate(graph_data.items()):
        definition = definitions[item["class_type"]]
        sockets, widgets, named = [], [], {}
        inputs = {**definition["input"].get("required", {}), **definition["input"].get("optional", {})}
        for key, entry in inputs.items():
            kind, settings = entry[0], entry[1] if len(entry)>1 else {}
            if key not in item["inputs"] and kind == "COMFY_DYNAMICCOMBO_V3":
                continue
            value = item["inputs"].get(key, settings.get("default"))
            connected = isinstance(value, list) and len(value)==2 and isinstance(value[0], str) and value[0] in graph_data
            is_widget = isinstance(kind, list) or kind in ("STRING","INT","FLOAT","BOOLEAN","COMBO","COMFY_DYNAMICCOMBO_V3")
            if is_widget and not settings.get("forceInput"):
                widgets.append(value)
                named[key] = value
                if settings.get("control_after_generate"):
                    widgets.append("fixed")
            if connected or not is_widget or settings.get("forceInput"):
                socket = {"name": key, "type": "COMBO" if isinstance(kind,list) else kind, "link": None}
                if is_widget:
                    socket["widget"] = {"name": key}
                sockets.append(socket)
        if item["class_type"] == "SaveVideo":
            widgets = [item["inputs"]["filename_prefix"], "mp4", "auto", "auto"]
            named = {"filename_prefix":widgets[0], "format":"mp4", "format.codec":"auto", "codec":"auto"}
        outputs = [{"name": n, "type": t, "links": []} for n,t in zip(definition.get("output_name", definition["output"]),definition["output"])]
        node = {"id":int(identity), "type":item["class_type"], "pos":positions[int(identity)],
                "size":[320,400 if identity=="6" else 230], "flags":{}, "order":order, "mode":0,
                "inputs":sockets, "outputs":outputs, "properties":{"Node name for S&R":item["class_type"],"cnr_id":"comfy-core"},
                "widgets_values":widgets, "widgets_values_named":named}
        nodes.append(node)
        by_id[identity]=node
    for identity,item in graph_data.items():
        for slot,socket in enumerate(by_id[identity]["inputs"]):
            value=item["inputs"].get(socket["name"])
            if isinstance(value,list) and len(value)==2 and value[0] in by_id:
                origin=by_id[value[0]]
                link_id=len(links)+1
                links.append([link_id,origin["id"],value[1],int(identity),slot,origin["outputs"][value[1]]["type"]])
                origin["outputs"][value[1]]["links"].append(link_id)
                socket["link"]=link_id
    return {"version":0.4,"last_node_id":max(n["id"] for n in nodes),"last_link_id":len(links),
            "nodes":nodes,"links":links,"groups":[],"config":{},"extra":{"ds":{"scale":0.45,"offset":[20,20]}}}


def queue(args):
    run=RUNS/args.name
    config=json.loads((run/"run.json").read_text(encoding="utf-8"))
    if config.get("prompt_id"):
        raise RuntimeError("Already queued; collect this job instead of duplicating it")
    if args.guide:
        guide_name=f"h3_graybox_{args.name}_guide.mp4"
        shutil.copy2(args.guide, COMFY/"input"/guide_name)
        config["guide"]=guide_name
        config["guide_padding"]={"right":(-config["width"])%32,"bottom":(-config["height"])%32,"main_canvas_unchanged":True}
    prompt=graph(config,args.name)
    ui=workflow(prompt,request("GET","/object_info"))
    save_json(run/"workflow_api.json",prompt)
    save_json(run/"workflow.json",ui)
    target=ROOT/"workflows"/f"minimax_h3_controlled_graybox_{args.name}.json"
    shutil.copy2(run/"workflow.json",target)
    shutil.copy2(target,COMFY/"user/default/workflows"/target.name)
    result=request("POST","/prompt",{"prompt":prompt,"extra_data":{"extra_pnginfo":{"workflow":ui}}})
    config.update(status="queued",prompt_id=result["prompt_id"],model_patch=PATCH)
    save_json(run/"run.json",config)
    print(json.dumps(result),flush=True)


def collect(args):
    run=RUNS/args.name
    config=json.loads((run/"run.json").read_text(encoding="utf-8"))
    item=request("GET","/history/"+config["prompt_id"]).get(config["prompt_id"])
    if not item:
        queue_state = request("GET","/queue")
        print(json.dumps({key:[item[1] for item in values] for key,values in queue_state.items()}),flush=True)
        return
    save_json(run/"history.json",item)
    if item["status"]["status_str"] != "success":
        config["status"]="failed"
        save_json(run/"run.json",config)
        raise RuntimeError(json.dumps(item["status"],ensure_ascii=False))
    entries=[entry for node in item["outputs"].values() for key in ("videos","images","gifs")
             for entry in node.get(key,[]) if entry.get("type") == "output" and entry["filename"].endswith(".mp4")]
    if not entries:
        raise RuntimeError("No saved MP4 in history")
    entry=entries[0]
    generated=COMFY/"output"/entry.get("subfolder","")/entry["filename"]
    shutil.copy2(generated,run/"comfy_raw.mp4")
    raw_probe=inspect(run/"comfy_raw.mp4")
    if raw_probe["frames"] != config["length"]:
        raise RuntimeError("ComfyUI returned the wrong number of generated frames")
    # Native video encoding rounds PTS to milliseconds. Re-encode the same
    # decoded frames with exact rational PTS, without sampling or interpolation.
    exact=Writer(run/"generated.mp4",config["width"],config["height"],60,crf=18)
    with av.open(str(generated)) as container:
        for frame in container.decode(video=0):
            exact.write(frame.to_ndarray(format="rgb24"))
    exact.close()
    probe=inspect(run/"generated.mp4")
    if not exact_timeline(probe,config["length"]):
        raise RuntimeError("Exact source-index encoding failed")
    config.update(status="generated_not_visually_approved",output=str(run/"generated.mp4"),probe=probe,comfy_raw_probe=raw_probe)
    save_json(run/"run.json",config)
    comparison=Writer(run/"comparison.mp4",config["width"]*2,config["height"],60)
    sheet=Image.new("RGB",(config["width"]*2,config["height"]*3))
    wanted={0:0,config["real_count"]//2:1,config["real_count"]-1:2}
    with av.open(str(SOURCE)) as src,av.open(str(generated)) as dst:
        source_frames=src.decode(video=0)
        for _ in range(config["start"]):
            next(source_frames)
        for index,frame in enumerate(dst.decode(video=0)):
            if index>=config["real_count"]:
                break
            rgb=cv2.resize(next(source_frames).to_ndarray(format="rgb24"),(config["width"],config["height"]))
            styled=frame.to_ndarray(format="rgb24")
            combined=np.concatenate((rgb,styled),axis=1)
            comparison.write(combined)
            if index in wanted:
                panel=Image.fromarray(combined)
                ImageDraw.Draw(panel).text((10,10),f"source frame {config['start']+index} | H3 graybox",fill="white")
                sheet.paste(panel,(0,wanted[index]*config["height"]))
    comparison.close()
    sheet.save(run/"contact_sheet.jpg",quality=92)
    print(f"Generated: {run/'generated.mp4'}; {probe['frames']} indexed frames. Visual approval pending.",flush=True)


def guide(args):
    run = RUNS/args.name
    config = json.loads((run/"run.json").read_text(encoding="utf-8"))
    path = run/"overlap_22.mp4"
    if path.exists():
        raise FileExistsError(path)
    writer = Writer(path, config["width"], config["height"], 24, crf=0)
    with av.open(str(run/"generated.mp4")) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index >= config["length"]-22:
                writer.write(frame.to_ndarray(format="rgb24"))
    writer.close()
    if inspect(path)["frames"] != 22:
        raise RuntimeError("Overlap guide must contain exactly 22 frames")
    print(path,flush=True)


def selected_mapping(configs):
    selected = []
    for part, config in enumerate(configs):
        for item in config["mapping"]:
            if item["padding"] or (part and item["source_index"] < len(selected)):
                continue
            if item["source_index"] != len(selected) or Fraction(item["source_pts_seconds"]) != Fraction(len(selected),60):
                raise ValueError("Gap or timestamp mismatch in source-index assembly")
            selected.append({**item, "part": part, "prompt_id": config["prompt_id"]})
    if len(selected) != 720:
        raise ValueError(f"Expected all 720 source frames, got {len(selected)}")
    return selected


def assemble(args):
    if not args.parts:
        raise ValueError("Specify chunk run names with --parts")
    configs = [json.loads((RUNS/name/"run.json").read_text(encoding="utf-8")) for name in args.parts]
    mapping = selected_mapping(configs)
    if any(c["source_sha256"] != configs[0]["source_sha256"] for c in configs):
        raise ValueError("Chunks use different source videos")
    run = RUNS/args.name
    run.mkdir(parents=True,exist_ok=False)
    writer = Writer(run/"graybox.mp4",1920,1080,60,crf=18)
    comparison = Writer(run/"comparison.mp4",1536,432,60,crf=18)
    kept = {(item["part"],item["generated_index"]): index for index,item in enumerate(mapping)}
    sheet = Image.new("RGB",(1536,432*6))
    keyframes = {index:row for row,index in enumerate((0,120,240,360,600,719))}
    changes, luma_means, blank_frames, previous_gray = [], [], [], None
    with av.open(str(SOURCE)) as source:
        source_frames = source.decode(video=0)
        for part,name in enumerate(args.parts):
            with av.open(str(RUNS/name/"generated.mp4")) as container:
                for local,frame in enumerate(container.decode(video=0)):
                    if (part,local) not in kept:
                        continue
                    index = kept[(part,local)]
                    styled = frame.to_ndarray(format="rgb24")
                    gray = cv2.cvtColor(cv2.resize(styled,(384,216)),cv2.COLOR_RGB2GRAY)
                    luma_means.append(float(np.mean(gray)))
                    if np.std(gray)<1 or np.mean(gray)<2:
                        blank_frames.append(index)
                    changes.append(0.0 if previous_gray is None else float(np.mean(cv2.absdiff(gray,previous_gray))))
                    previous_gray = gray
                    writer.write(cv2.resize(styled,(1920,1080),interpolation=cv2.INTER_CUBIC))
                    original = next(source_frames).to_ndarray(format="rgb24")
                    pair = np.concatenate((cv2.resize(original,(768,432)),cv2.resize(styled,(768,432))),axis=1)
                    comparison.write(pair)
                    if index in keyframes:
                        panel = Image.fromarray(pair)
                        ImageDraw.Draw(panel).text((10,10),f"source frame {index} | H3 graybox",fill="white")
                        sheet.paste(panel,(0,keyframes[index]*432))
    writer.close()
    comparison.close()
    sheet.save(run/"contact_sheet.jpg",quality=92)
    probe = inspect(run/"graybox.mp4")
    failure_path = RUNS/"full_failures.json"
    failures = json.loads(failure_path.read_text(encoding="utf-8"))["failures"] if failure_path.exists() else []
    seams = [i for i in range(1,len(mapping)) if mapping[i]["part"] != mapping[i-1]["part"]]
    temporal_ok = exact_timeline(probe,720)
    save_json(run/"report.json",{"schema":"h3.graybox-delivery/1", "source":str(SOURCE),
        "source_sha256":configs[0]["source_sha256"],"parts":args.parts,"mapping":mapping,
        "parameters":[{key:value for key,value in c.items() if key not in ("mapping","probe","comfy_raw_probe")} for c in configs],
        "models":json.loads((RUNS/"models.json").read_text(encoding="utf-8")),
        "execution_failures":[f for f in failures if f.get("retry") in args.parts or f.get("run") in args.parts],
        "frame_checks":{"blank_frame_indices":blank_frames,"luma_mean_range":[min(luma_means),max(luma_means)],
                        "adjacent_luma_mae_median":float(np.median(changes[1:])),
                        "adjacent_luma_mae_p95":float(np.percentile(changes[1:],95)),
                        "seams":[{"source_index":i,"luma_mae":changes[i]} for i in seams],
                        "largest_changes":[{"source_index":int(i),"luma_mae":changes[i]} for i in np.argsort(changes)[-5:][::-1]],
                        "note":"Frame variation is a diagnostic, not a proof of motion fidelity or absence of local artifacts."},
        "probe":probe,"duration_seconds":12,"temporal_pass":temporal_ok,
        "motion_style_pass":False,"status":"assembled_visual_review_pending",
        "spatial_resize":"Native H3 frames resized to 1920x1080; no temporal interpolation or frame synthesis during assembly.",
        "limitations":configs[0]["limitations"]})
    if not temporal_ok:
        raise RuntimeError("Final frame count/FPS/PTS verification failed")
    print(f"Assembled {run/'graybox.mp4'}: 720 frames, 60 FPS, final PTS {probe['pts_seconds'][-1]}",flush=True)


def verify(args):
    run = RUNS/args.name
    report = json.loads((run/"report.json").read_text(encoding="utf-8"))
    probe, comparison = inspect(run/"graybox.mp4"), inspect(run/"comparison.mp4")
    mapping_ok = len(report["mapping"]) == 720 and all(
        item["source_index"] == i and not item["padding"] and Fraction(item["source_pts_seconds"]) == Fraction(i,60)
        for i,item in enumerate(report["mapping"]))
    passed = exact_timeline(probe,720) and exact_timeline(comparison,720) and mapping_ok
    report.update(probe=probe,comparison_probe=comparison,temporal_pass=passed,
                  source_mapping_pass=mapping_ok,status="complete_source_indexed_approximation" if passed else "verification_failed",
                  user_accepted_preview="pilot_canny_v3_fixed",strict_motion_geometry_pass=False,
                  visible_limitations=["Generated layouts and fine foliage may differ from source, as in the accepted preview.",
                                       "Visible scene changes at source frames 260 (4.3333s) and 498 (8.3s)."],
                  output_sha256=sha256(run/"graybox.mp4"),comparison_sha256=sha256(run/"comparison.mp4"))
    save_json(run/"report.json",report)
    if not passed:
        raise RuntimeError("Final decoded timeline or source mapping failed")
    print(f"Verified {run/'graybox.mp4'}: 720 frames, 60 FPS, 12 seconds; mapping 0..719.",flush=True)


async def watch_events():
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(SERVER.replace("http", "ws", 1)+"/ws") as ws:
            deadline = asyncio.get_running_loop().time()+45
            while asyncio.get_running_loop().time()<deadline:
                try:
                    message = await ws.receive(timeout=10)
                except asyncio.TimeoutError:
                    continue
                if message.type == aiohttp.WSMsgType.TEXT:
                    event = json.loads(message.data)
                    if event["type"] in ("status","executing","progress","progress_text","execution_error","execution_success"):
                        print(json.dumps(event),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("prepare","queue","collect","watch","guide","assemble","verify"))
    parser.add_argument("--name",required=True)
    parser.add_argument("--start",type=int,default=120)
    parser.add_argument("--length",type=int,default=124)
    parser.add_argument("--width",type=int,default=768)
    parser.add_argument("--height",type=int,default=432)
    parser.add_argument("--mode",choices=("gray","canny"),default="gray")
    parser.add_argument("--steps",type=int,default=8)
    parser.add_argument("--no-turbo",action="store_true")
    parser.add_argument("--strength",type=float,default=0.85)
    parser.add_argument("--seed",type=int,default=66095357)
    parser.add_argument("--guide",type=Path)
    parser.add_argument("--parts",nargs="+")
    args=parser.parse_args()
    if not args.name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in args.name):
        parser.error("name must use lowercase letters, digits, hyphens or underscores")
    if args.action == "watch":
        asyncio.run(watch_events())
    else:
        {"prepare":prepare,"queue":queue,"collect":collect,"guide":guide,"assemble":assemble,"verify":verify}[args.action](args)


if __name__=="__main__":
    main()
