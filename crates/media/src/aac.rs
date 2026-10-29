//! The AudioSpecificConfig a movie's first AAC track decodes with, from the containers that carry
//! it outside the audio frames: MP4 / MOV (`esds`), Matroska (`CodecPrivate`) and LATM in MPEG
//! TS / PS (its StreamMuxConfig). FilmCraft's other AAC is ADTS (in MPEG TS / PS or a raw `.aac`),
//! whose channel configurations FilmCraft decodes (1 to 7) its channel count tells apart
//! ([`crate::downmix::Order::Aac`]).

use std::io;

use filmcraft_aac::AudioSpecificConfig;
use filmcraft_media::ByteReader;

/// How far into an MPEG TS / PS file its first LATM frame is looked for: the stream is indexed
/// as far as this, not through the whole file.
const MPEG_HEAD: u64 = 8 << 20;

/// The configuration of the first audio track, if it is AAC: from an MP4 / MOV `header` (its
/// ftyp and moov, or the whole file), or else from `file`, read as Matroska or MPEG TS / PS.
pub(crate) fn config(header: Option<&[u8]>, file: Option<&dyn ByteReader>) -> Option<AudioSpecificConfig> {
    let asc = match header.and_then(mp4) {
        Some(asc) => asc,
        None => {
            let file = file?;
            let mut magic = [0u8; 4];
            file.read_at(0, &mut magic).ok()?;
            if magic == [0x1A, 0x45, 0xDF, 0xA3] { matroska(file)? } else { latm(file)? }
        }
    };
    AudioSpecificConfig::parse(&asc).ok()
}

fn mp4(bytes: &[u8]) -> Option<Vec<u8>> {
    let file = filmcraft_isobmff::open(bytes).ok()?;
    let track = file.tracks.iter().find(|t| t.kind == filmcraft_isobmff::TrackKind::Audio)?;
    match &track.entries.first()?.codec {
        filmcraft_isobmff::CodecConfig::Aac(a) => Some(a.asc.clone()),
        _ => None,
    }
}

fn matroska(file: &dyn ByteReader) -> Option<Vec<u8>> {
    let opts = filmcraft_matroska::OpenOptions { index: false, ..Default::default() };
    let mkv = filmcraft_matroska::open_with(Head(file, file.len()), &opts).ok()?;
    match &mkv.tracks.iter().find(|t| t.kind == filmcraft_matroska::TrackKind::Audio)?.codec {
        filmcraft_matroska::Codec::Aac { asc } => Some(asc.clone()),
        _ => None,
    }
}

fn latm(file: &dyn ByteReader) -> Option<Vec<u8>> {
    let head = Head(file, file.len().min(MPEG_HEAD));
    let mpeg = filmcraft_mpegts::open(&head).ok()?;
    let stream = mpeg.find(filmcraft_mpegts::Kind::Audio, |c| *c == filmcraft_mpegts::Codec::AacLatm)?;
    let frame = mpeg.read_unit(&head, stream, 0).ok()?;
    Some(filmcraft_codecs::audio::latm_config(&frame)?.asc)
}

/// The first `.1` bytes of a file, as the byte source FilmCraft's demuxers read.
struct Head<'a>(&'a dyn ByteReader, u64);

impl Head<'_> {
    fn read(&self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        match offset.checked_add(buf.len() as u64) {
            Some(end) if end <= self.1 => self.0.read_at(offset, buf),
            _ => Err(io::Error::new(io::ErrorKind::UnexpectedEof, "read past the end")),
        }
    }
}

impl filmcraft_matroska::ByteSource for Head<'_> {
    fn len(&self) -> u64 {
        self.1
    }
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        self.read(offset, buf)
    }
}

impl filmcraft_mpegts::ByteSource for Head<'_> {
    fn len(&self) -> u64 {
        self.1
    }
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        self.read(offset, buf)
    }
}
