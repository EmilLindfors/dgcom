//! Particle files: the particles of a run frame by frame, next to its snapshot file
//! (`io::snapshot`) for replays and for connectivity statistics between farms.
//!
//! A frame holds every particle released so far, in a fixed order (particle `i` is
//! the same particle in every frame that has it, so frames only grow): its position,
//! its height (the surface at the particle in 2D, its own height in 3D, m), its
//! status ([`status_code`]), its kind (an index into the header's kinds) and its
//! release time. Frames differ in length, so each starts with its byte length; the
//! reader indexes them on opening and ignores a partial last one, as a run still
//! writing leaves it.
//!
//! # Format (little-endian)
//!
//! ```text
//! magic        8 bytes   b"DGPART\0\x01" (format version 1)
//! header_len   u64       bytes of the header that follows
//! header:
//!   kinds                u64 bytes of UTF-8, one kind name per line
//!   metadata             u64 bytes of UTF-8, `key=value` lines
//! frames, each:
//!   frame_len            u64, bytes of the rest of the frame
//!   t                    f64 (model time, s)
//!   n                    u64, particles
//!   x, y, z              f32 × n each (m)
//!   status, kind         u8 × n each
//!   born                 f32 × n (model time of release, s)
//! ```

use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

use crate::io::SnapshotError;
use crate::particles::ParticleStatus;

const MAGIC: &[u8; 8] = b"DGPART\0\x01";
/// Bytes per particle in a frame.
const PARTICLE_BYTES: usize = 3 * 4 + 2 + 4;

/// A particle's status as stored: 0 active, 1 stranded, 2 out through an open
/// boundary, 3 settled on the bed, 4 dead.
pub fn status_code(status: ParticleStatus) -> u8 {
    match status {
        ParticleStatus::Active => 0,
        ParticleStatus::Stranded => 1,
        ParticleStatus::Exited(_) => 2,
        ParticleStatus::Settled => 3,
        ParticleStatus::Dead => 4,
    }
}

/// Every particle at one instant, in release order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ParticleFrame {
    /// Model time (s)
    pub t: f64,
    /// Position, mesh coordinates (m)
    pub xy: Vec<[f32; 2]>,
    /// Height (m): the surface at the particle in 2D, the particle's own in 3D
    pub z: Vec<f32>,
    /// [`status_code`]
    pub status: Vec<u8>,
    /// Index into the file's kinds
    pub kind: Vec<u8>,
    /// Model time of release (s)
    pub born: Vec<f32>,
}

impl ParticleFrame {
    pub fn len(&self) -> usize {
        self.xy.len()
    }

    pub fn is_empty(&self) -> bool {
        self.xy.is_empty()
    }

    fn check(&self) -> Result<(), SnapshotError> {
        let n = self.len();
        if [
            self.z.len(),
            self.status.len(),
            self.kind.len(),
            self.born.len(),
        ]
        .iter()
        .any(|&l| l != n)
        {
            return Err(SnapshotError::Format(format!(
                "a particle frame's fields differ in length ({n} positions)"
            )));
        }
        Ok(())
    }
}

/// Writes a particle file: the header on creation, then one frame per call.
pub struct ParticleFileWriter {
    out: BufWriter<File>,
    n_kinds: usize,
    bytes: Vec<u8>,
}

impl ParticleFileWriter {
    /// Create `path` for particles of the given kinds (names without newlines).
    pub fn create(
        path: impl AsRef<Path>,
        kinds: &[&str],
        metadata: &[(&str, &str)],
    ) -> Result<Self, SnapshotError> {
        if kinds.is_empty() || kinds.len() > 256 || kinds.iter().any(|k| k.contains('\n')) {
            return Err(SnapshotError::Format(
                "1 to 256 kinds, names without newlines".into(),
            ));
        }
        if metadata
            .iter()
            .any(|(k, v)| k.contains(['=', '\n']) || v.contains('\n'))
        {
            return Err(SnapshotError::Format(
                "metadata: no '=' in keys, no newlines".into(),
            ));
        }
        let mut header = Vec::new();
        for text in [
            kinds.join("\n"),
            metadata.iter().map(|(k, v)| format!("{k}={v}\n")).collect(),
        ] {
            header.extend_from_slice(&(text.len() as u64).to_le_bytes());
            header.extend_from_slice(text.as_bytes());
        }
        let mut out = BufWriter::new(File::create(path)?);
        out.write_all(MAGIC)?;
        out.write_all(&(header.len() as u64).to_le_bytes())?;
        out.write_all(&header)?;
        out.flush()?;
        Ok(Self {
            out,
            n_kinds: kinds.len(),
            bytes: Vec::new(),
        })
    }

    /// Append `frame`, flushed whole.
    pub fn write(&mut self, frame: &ParticleFrame) -> Result<(), SnapshotError> {
        frame.check()?;
        if let Some(&k) = frame.kind.iter().find(|&&k| k as usize >= self.n_kinds) {
            return Err(SnapshotError::Format(format!(
                "kind {k} of a file with {} kinds",
                self.n_kinds
            )));
        }
        let n = frame.len();
        let b = &mut self.bytes;
        b.clear();
        b.extend_from_slice(&((16 + PARTICLE_BYTES * n) as u64).to_le_bytes());
        b.extend_from_slice(&frame.t.to_le_bytes());
        b.extend_from_slice(&(n as u64).to_le_bytes());
        for c in 0..2 {
            frame
                .xy
                .iter()
                .for_each(|p| b.extend_from_slice(&p[c].to_le_bytes()));
        }
        frame
            .z
            .iter()
            .for_each(|z| b.extend_from_slice(&z.to_le_bytes()));
        b.extend_from_slice(&frame.status);
        b.extend_from_slice(&frame.kind);
        frame
            .born
            .iter()
            .for_each(|t| b.extend_from_slice(&t.to_le_bytes()));
        self.out.write_all(&self.bytes)?;
        self.out.flush()?;
        Ok(())
    }
}

/// Reads a particle file: the frames are indexed on opening.
pub struct ParticleFileReader {
    file: File,
    /// The particles' kinds and the file's metadata
    pub kinds: Vec<String>,
    pub metadata: Vec<(String, String)>,
    /// Offset and time of every complete frame
    frames: Vec<(u64, f64)>,
}

impl ParticleFileReader {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, SnapshotError> {
        let mut file = File::open(path)?;
        let mut start = [0u8; 16];
        file.read_exact(&mut start)
            .map_err(|_| SnapshotError::Format("shorter than its magic number".into()))?;
        if &start[..8] != MAGIC {
            return Err(SnapshotError::Format(
                "wrong magic number (not a dgcom particle file, or another version)".into(),
            ));
        }
        let header_len = u64::from_le_bytes(start[8..].try_into().unwrap()) as usize;
        let mut header = vec![0u8; header_len];
        file.read_exact(&mut header)
            .map_err(|_| SnapshotError::Format("truncated header".into()))?;
        let mut texts = Vec::new();
        let mut at = 0;
        for _ in 0..2 {
            let truncated = || SnapshotError::Format("truncated header".into());
            let len = header
                .get(at..at + 8)
                .ok_or_else(truncated)
                .map(|b| u64::from_le_bytes(b.try_into().unwrap()) as usize)?;
            let text = header.get(at + 8..at + 8 + len).ok_or_else(truncated)?;
            texts.push(
                String::from_utf8(text.to_vec())
                    .map_err(|_| SnapshotError::Format("header is not UTF-8".into()))?,
            );
            at += 8 + len;
        }
        let kinds = texts[0].lines().map(str::to_string).collect();
        let metadata = texts[1]
            .lines()
            .filter_map(|l| l.split_once('='))
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        let mut reader = Self {
            file,
            kinds,
            metadata,
            frames: Vec::new(),
        };
        reader.index(16 + header_len as u64)?;
        Ok(reader)
    }

    /// Index the complete frames from `offset` on.
    fn index(&mut self, mut offset: u64) -> Result<(), SnapshotError> {
        let len = self.file.metadata()?.len();
        let mut head = [0u8; 16];
        while offset + 16 <= len {
            self.file.seek(SeekFrom::Start(offset))?;
            self.file.read_exact(&mut head)?;
            let frame_len = u64::from_le_bytes(head[..8].try_into().unwrap());
            if offset + 8 + frame_len > len {
                break;
            }
            self.frames
                .push((offset, f64::from_le_bytes(head[8..].try_into().unwrap())));
            offset += 8 + frame_len;
        }
        Ok(())
    }

    /// Complete frames indexed.
    pub fn n_frames(&self) -> usize {
        self.frames.len()
    }

    /// Model time of frame `i`.
    pub fn time(&self, i: usize) -> f64 {
        self.frames[i].1
    }

    /// Frame `i`.
    pub fn read_frame(&mut self, i: usize) -> Result<ParticleFrame, SnapshotError> {
        let (offset, t) = self.frames[i];
        self.file.seek(SeekFrom::Start(offset + 16))?;
        let mut count = [0u8; 8];
        self.file.read_exact(&mut count)?;
        let n = u64::from_le_bytes(count) as usize;
        let mut bytes = vec![0u8; PARTICLE_BYTES * n];
        self.file.read_exact(&mut bytes)?;
        let f32s = |b: &[u8]| -> Vec<f32> {
            b.as_chunks::<4>()
                .0
                .iter()
                .map(|&c| f32::from_le_bytes(c))
                .collect()
        };
        let (x, y, z) = (
            f32s(&bytes[..4 * n]),
            f32s(&bytes[4 * n..8 * n]),
            f32s(&bytes[8 * n..12 * n]),
        );
        let frame = ParticleFrame {
            t,
            xy: x.into_iter().zip(y).map(|(x, y)| [x, y]).collect(),
            z,
            status: bytes[12 * n..13 * n].to_vec(),
            kind: bytes[13 * n..14 * n].to_vec(),
            born: f32s(&bytes[14 * n..18 * n]),
        };
        if let Some(&k) = frame.kind.iter().find(|&&k| k as usize >= self.kinds.len()) {
            return Err(SnapshotError::Format(format!(
                "frame {i}: kind {k} of a file with {} kinds",
                self.kinds.len()
            )));
        }
        Ok(frame)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(t: f64, n: usize) -> ParticleFrame {
        ParticleFrame {
            t,
            xy: (0..n).map(|i| [i as f32, -(i as f32)]).collect(),
            z: (0..n).map(|i| -0.5 * i as f32).collect(),
            status: (0..n).map(|i| (i % 5) as u8).collect(),
            kind: (0..n).map(|i| (i % 3) as u8).collect(),
            born: (0..n).map(|i| 60.0 * i as f32).collect(),
        }
    }

    #[test]
    fn particles_read_back_as_written() {
        let path = std::env::temp_dir().join(format!("dgcom-particles-{}", std::process::id()));
        let mut writer = ParticleFileWriter::create(
            &path,
            &["lice larvae", "faeces", "feed"],
            &[("source", "test")],
        )
        .unwrap();
        // Frames grow; an empty one is allowed
        let frames = [frame(0.0, 0), frame(60.0, 4), frame(120.0, 9)];
        for f in &frames {
            writer.write(f).unwrap();
        }
        // A partial frame is not counted
        writer
            .out
            .write_all(&[200, 0, 0, 0, 0, 0, 0, 0, 1, 2])
            .unwrap();
        writer.out.flush().unwrap();

        let mut reader = ParticleFileReader::open(&path).unwrap();
        assert_eq!(reader.kinds, ["lice larvae", "faeces", "feed"]);
        assert_eq!(reader.metadata, [("source".into(), "test".into())]);
        assert_eq!(reader.n_frames(), 3);
        assert_eq!(reader.time(2), 120.0);
        for (i, f) in frames.iter().enumerate() {
            assert_eq!(&reader.read_frame(i).unwrap(), f);
        }
        // Kinds the file does not have, and fields of different lengths, are refused
        let mut bad = frame(180.0, 4);
        bad.kind[1] = 3;
        assert!(writer.write(&bad).is_err());
        bad.kind[1] = 0;
        bad.born.pop();
        assert!(writer.write(&bad).is_err());
        assert_eq!(status_code(ParticleStatus::Dead), 4);
        std::fs::remove_file(&path).unwrap();
    }
}
