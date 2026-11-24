"""Minimal ISO-BMFF sample-table inspection; no full-video download or decoding."""
import struct

from stage_a.access import InspectionError


def boxes(data, start=0, end=None):
    end = len(data) if end is None else end
    cursor = start
    while cursor < end:
        if cursor+8 > end:
            raise InspectionError("Truncated MP4 box header")
        size, kind = struct.unpack_from(">I4s", data, cursor)
        header = 8
        if size == 1:
            if cursor+16 > end:
                raise InspectionError("Truncated large MP4 box")
            size = struct.unpack_from(">Q", data, cursor+8)[0]
            header = 16
        elif size == 0:
            size = end-cursor
        if size < header or cursor+size > end:
            raise InspectionError("Invalid MP4 box size")
        yield kind, cursor, cursor+header, cursor+size
        cursor += size


def child(data, parent, kind):
    matches = [box for box in boxes(data, parent[2], parent[3]) if box[0] == kind]
    if len(matches) != 1:
        raise InspectionError(f"Expected one MP4 {kind.decode()} box")
    return matches[0]


def tracks(data):
    root = list(boxes(data))[0]
    return [box for box in boxes(data, root[2], root[3]) if box[0] == b"trak"]


def video_prefix_end(moov, sample_count):
    for track in tracks(moov):
        mdia = child(moov, track, b"mdia")
        handler = child(moov, mdia, b"hdlr")
        if moov[handler[2]+8:handler[2]+12] != b"vide":
            continue
        stbl = child(moov, child(moov, mdia, b"minf"), b"stbl")
        children = {box[0]: box for box in boxes(moov, stbl[2], stbl[3])}
        co = children.get(b"co64") or children.get(b"stco")
        sc, sz = children[b"stsc"], children[b"stsz"]
        if co is None:
            raise InspectionError("No MP4 chunk offsets")
        offset_count = struct.unpack_from(">I", moov, co[2]+4)[0]
        fmt, stride = (">Q", 8) if co[0] == b"co64" else (">I", 4)
        offsets = [struct.unpack_from(fmt, moov, co[2]+8+i*stride)[0] for i in range(offset_count)]
        entries = [struct.unpack_from(">III", moov, sc[2]+8+12*i) for i in range(struct.unpack_from(">I", moov, sc[2]+4)[0])]
        fixed_size, total = struct.unpack_from(">II", moov, sz[2]+4)
        if not 0 < sample_count <= total:
            raise InspectionError("Requested MP4 sample window is out of bounds")
        sizes = [fixed_size]*total if fixed_size else [struct.unpack_from(">I", moov, sz[2]+12+4*i)[0] for i in range(total)]
        samples = 0
        rule = 0
        end = 0
        for index, offset in enumerate(offsets, 1):
            while rule+1 < len(entries) and entries[rule+1][0] <= index:
                rule += 1
            in_chunk = entries[rule][1]
            end = max(end, offset+sum(sizes[samples:samples+in_chunk]))
            samples += in_chunk
            if samples >= sample_count:
                return end, {"video_sample_count": total, "prefix_video_samples_including_preroll": samples,
                             "source_chunk_offset_layout": co[0].decode()}
    raise InspectionError("No video track in MP4")


def relocate_chunks(moov, delta):
    patched = bytearray(moov)
    for track in tracks(moov):
        stbl = child(moov, child(moov, child(moov, track, b"mdia"), b"minf"), b"stbl")
        for box in boxes(moov, stbl[2], stbl[3]):
            if box[0] not in (b"stco", b"co64"):
                continue
            count = struct.unpack_from(">I", moov, box[2]+4)[0]
            fmt, stride = (">Q", 8) if box[0] == b"co64" else (">I", 4)
            for index in range(count):
                position = box[2]+8+index*stride
                shifted = struct.unpack_from(fmt, moov, position)[0]+delta
                if shifted < 0 or (stride == 4 and shifted >= 2**32):
                    raise InspectionError("Relocated MP4 offset does not fit original table")
                struct.pack_into(fmt, patched, position, shifted)
    return bytes(patched)
