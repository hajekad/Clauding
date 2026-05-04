//! Minimal PNG writer — no crate dependencies.
//! Uncompressed deflate (store blocks), Adler-32 checksum, CRC-32.

pub fn save_png(pixels: &[u32], w: usize, h: usize, path: &str) {
    use std::io::Write;

    let row_bytes = 1 + w * 3;
    let mut raw = Vec::with_capacity(row_bytes * h);
    for y in 0..h {
        raw.push(0u8);
        for x in 0..w {
            let c = pixels[y * w + x];
            raw.push(((c >> 16) & 0xFF) as u8);
            raw.push(((c >> 8) & 0xFF) as u8);
            raw.push((c & 0xFF) as u8);
        }
    }

    let mut deflate = Vec::with_capacity(raw.len() + raw.len() / 65535 * 5 + 20);
    deflate.push(0x78);
    deflate.push(0x01);

    let mut offset = 0;
    while offset < raw.len() {
        let remaining = raw.len() - offset;
        let block_len = remaining.min(65535);
        let is_last = offset + block_len >= raw.len();
        deflate.push(if is_last { 1 } else { 0 });
        deflate.push((block_len & 0xFF) as u8);
        deflate.push(((block_len >> 8) & 0xFF) as u8);
        deflate.push((!block_len & 0xFF) as u8);
        deflate.push(((!block_len >> 8) & 0xFF) as u8);
        deflate.extend_from_slice(&raw[offset..offset + block_len]);
        offset += block_len;
    }

    let (mut s1, mut s2): (u32, u32) = (1, 0);
    for &b in &raw {
        s1 = (s1 + b as u32) % 65521;
        s2 = (s2 + s1) % 65521;
    }
    deflate.extend_from_slice(&((s2 << 16) | s1).to_be_bytes());

    fn crc32(data: &[u8]) -> u32 {
        let mut crc: u32 = 0xFFFFFFFF;
        for &b in data {
            crc ^= b as u32;
            for _ in 0..8 {
                crc = if crc & 1 != 0 {
                    (crc >> 1) ^ 0xEDB88320
                } else {
                    crc >> 1
                };
            }
        }
        !crc
    }

    fn write_chunk(out: &mut Vec<u8>, tag: &[u8; 4], data: &[u8]) {
        out.extend_from_slice(&(data.len() as u32).to_be_bytes());
        out.extend_from_slice(tag);
        out.extend_from_slice(data);
        let mut crc_data = Vec::with_capacity(4 + data.len());
        crc_data.extend_from_slice(tag);
        crc_data.extend_from_slice(data);
        out.extend_from_slice(&crc32(&crc_data).to_be_bytes());
    }

    let mut png = Vec::with_capacity(deflate.len() + 100);
    png.extend_from_slice(&[137, 80, 78, 71, 13, 10, 26, 10]);

    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&(w as u32).to_be_bytes());
    ihdr.extend_from_slice(&(h as u32).to_be_bytes());
    ihdr.push(8);
    ihdr.push(2);
    ihdr.push(0);
    ihdr.push(0);
    ihdr.push(0);
    write_chunk(&mut png, b"IHDR", &ihdr);
    write_chunk(&mut png, b"IDAT", &deflate);
    write_chunk(&mut png, b"IEND", &[]);

    let mut f = std::fs::File::create(path).unwrap();
    f.write_all(&png).unwrap();
    eprintln!(
        "Saved: {} ({}x{}, {:.1}MB)",
        path,
        w,
        h,
        png.len() as f64 / 1_000_000.0
    );
}
